"""GPU rendering of the 3D bead scenes (the MesoMem patch, sheet and assembly
box) with moderngl.

The CPU path in renderer.py drew each bead as a numpy-shaded sphere sprite,
painter-sorted back to front -- which has no per-pixel depth, so beads close in z
"jump" in front of / behind each other instead of intersecting, and the ~900-bead
sheet regenerates sprites almost every frame. This module replaces that with the
standard molecular-viz technique, and then with the deferred post-processing
chain the standalone showreel (21_LearnModernGL/showreel.py) was tuned on.

  PASS 1  G-BUFFER      instanced sphere IMPOSTORS, one draw call. Each bead is
                        one camera-facing quad; the fragment shader ray-casts the
                        sphere and writes gl_FragDepth, so the hardware depth
                        buffer resolves intersections exactly (no sorting, no
                        z-jumps). Out: albedo, view normal, view position.
  PASS 2  OCCLUSION     half-res, two terms in one pass: ambient occlusion from
                        hemisphere sampling, and sun visibility from a march
                        toward the light through the depth buffer
  PASS 3  BLUR          foreground-masked box blur of both, which is what turns
                        their sampling noise into soft gradients
  PASS 4  COMPOSITE     lighting, outline, depth cue, tonemap
                        -> an offscreen colour texture
  PASS 5  DEPTH OF FIELD
  PASS 6  LINES         bonds / control net / box outline, as real depth-tested
                        GL lines (occluded by the beads for free), drawn AFTER
                        the blur so they stay crisp and un-tonemapped.
  PASS 7  FXAA          -> the final texture. Deferred rendering rules MSAA out,
                        and every edge here is a hard one, so the antialiasing
                        is a screen-space pass over the finished image.

Every pass after the first is screen-space: its cost depends on how many PIXELS
there are, not how many beads, so the 1500-bead assembly box post-processes for
the same price as the 7-bead patch.

What the picture looks like is not decided here -- see `lammps_live/render_style.py`
for the knobs and each playground file for its overrides.

The pipeline is context-agnostic: it takes a moderngl context (from the pygame GL
window, or a headless standalone context for benchmarking/tests) and renders into
its own framebuffers. `render()` fills `self.final_fbo`; `blit_to_viewport()`
copies it to the on-screen sim viewport, and `read_rgb()` reads it back (used by
the benchmark / snapshot tooling).

Requires an OpenGL 3.3+ core context (GLSL 330); works on macOS 4.1 core and Linux
via EGL/GLX.
"""
import numpy as np

from ..render_style import DEFAULT_STYLE
from .theme import (
    BEAD_BAND_HALFWIDTH, BEAD_BAND_SOFT, BEAD_EQUATOR_COLOR, BEAD_POLE_COLOR,
    BEAD_WHITE_POLE_COLOR, BEAD_WHITE_POLE_MIN, BEAD_WHITE_POLE_SOFT,
    BEND_RAMP,
    INFERNO,
)

# The circle-of-confusion radius in RenderStyle is quoted at this viewport
# height, and scaled from it, so fullscreen gets the same *picture* rather than
# the same pixel count of blur.
DOF_REFERENCE_HEIGHT = 900.0


# ---- small matrix helpers (row-major math; transposed on upload to GL) --------

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def view_matrix(eye, right, true_up, forward):
    """World->view (camera at origin looking down -Z, x right, y up), matching
    Camera3D's basis. Row-major; view = R*(world - eye)."""
    R = np.array([right, true_up, -forward], dtype=np.float64)
    t = -R @ np.asarray(eye, dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def proj_matrix(fx, fy, near, far):
    """Perspective projection consistent with Camera3D's pixel focal length:
    fx = focal/(W/2), fy = focal/(H/2). Maps view space (front = -z) to clip."""
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = fx
    M[1, 1] = fy
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -2.0 * far * near / (far - near)
    M[3, 2] = -1.0
    return M


def _gl(m):
    """Row-major numpy matrix -> column-major bytes for a GLSL mat4 uniform."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


def to_linear(color255, gamma):
    """A display-space 0..255 theme colour -> linear light, which is the space
    all the shading below works in (see render_style's module docstring)."""
    return tuple(float(c / 255.0) ** gamma for c in color255)


# ---- shader sources -----------------------------------------------------------

_GEOM_VS = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;
in vec2 in_corner;      // quad corner in [-1,1]
in vec3 in_center;      // per-instance world center
in float in_radius;     // per-instance world radius
in vec3 in_dir;         // per-instance unit director (world)
in float in_bright;     // per-instance albedo brightness multiplier (1 = normal)
in float in_energy;     // per-instance potential energy, for the energy colouring
in vec4 in_tint;        // per-instance flat albedo (rgb) + how much of it to use (a)
in float in_fade;       // per-instance 1 = full strength, 0 = fully faded to the background
in float in_material;   // 0 = an ordinary bead, else a body's own material (renderer.BODY_MATERIALS)
out vec3 v_centerView;
out float v_radius;
out vec3 v_dirView;
out vec3 v_rayView;
out float v_bright;
out float v_energy;
out vec4 v_tint;
out float v_fade;
out float v_material;
void main() {
    vec4 cv = view * vec4(in_center, 1.0);
    vec3 C = cv.xyz;
    float r = in_radius;
    v_centerView = C;
    v_radius = r;
    v_dirView = mat3(view) * in_dir;
    v_bright = in_bright;
    v_energy = in_energy;
    v_tint = in_tint;
    v_fade = in_fade;
    v_material = in_material;

    // ---- THE EXACT SILHOUETTE BILLBOARD ---------------------------------
    // A sphere's silhouette is a disc perpendicular to the direction TO THE
    // SPHERE, not to the screen, so off-axis it projects to an ELLIPSE that a
    // screen-parallel quad of radius r is too small to contain (8% short at 20
    // degrees off-axis, 36% at 40 -- and the corners of a wide viewport are out
    // past 30, which is where beads used to get visibly sliced by the edge of
    // their own quad; the old fix was a blanket 1.15x enlargement, which was
    // both too small out there and wasted fragments in the middle).
    //
    // Build the quad in the plane of the true silhouette instead: with
    // t = sqrt(d^2 - r^2) the distance to a tangent point, the tangent circle
    // has radius r*t/d and sits at t^2/d along the view direction. The quad
    // corners circumscribe the unit circle, so this covers it exactly from
    // every angle.
    float d = length(C);
    vec3 fwd = C / d;
    float t = sqrt(max(d * d - r * r, 1e-6));
    float rc = r * t / d;              // silhouette circle radius
    vec3 Cc = fwd * (t * t / d);       // silhouette circle centre
    vec3 hint = abs(fwd.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(hint, fwd));
    vec3 up = cross(fwd, right);
    vec3 corner = Cc + (in_corner.x * right + in_corner.y * up) * rc;

    v_rayView = corner;
    gl_Position = proj * vec4(corner, 1.0);
}
"""

_GEOM_FS = """
#version 330 core
uniform mat4 proj;
uniform mat4 viewInv;     // view->world, to clip the beads against the periodic box
uniform float band_half;
uniform float band_soft;
uniform vec3 equator_col;
uniform vec3 pole_col;
uniform vec3 white_col;
uniform float white_min;
uniform float white_soft;
uniform vec2 boxHalf;     // periodic-box half-extents in world x,y (centered at origin)
uniform float clipEnable; // 1 for periodic scenes (clip beads to the box), 0 otherwise
uniform float colorMode;  // 0 = director bands, 1 = energy colormap, 2 = per-bead tint
uniform vec2 energyRange; // (lo, hi) of the colormap, in the model's energy units
uniform vec3 ramp[32];    // the colormap, sampled (see theme.INFERNO)
uniform vec3 ramp2[32];   // the tinted species' own colormap (theme.BEND_RAMP)
uniform vec2 energyRange2; // ... and its range (RenderStyle.tint_energy_range)
uniform float tintEnergy; // 1 -> a tinted bead (mix > 0.5) is on ramp2 instead
uniform vec3 bodyLight;   // the mottling's two colours (style.body_color_light/dark),
uniform vec3 bodyDark;    // ... already in linear light
uniform float bodyMottle; // world size of the coarsest noise octave
in vec3 v_centerView;
in float v_radius;
in vec3 v_dirView;
in vec3 v_rayView;
in float v_bright;
in float v_energy;
in vec4 v_tint;
in float v_fade;
in float v_material;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec3 o_normal;
layout(location=2) out vec4 o_viewpos;

// ---- value noise, for the bacterium material -------------------------------
// A hash-per-lattice-point 3D value noise, trilinearly interpolated, summed over
// three halving octaves. Cheap (no gradients, no permutation table) and this is
// not asking for Perlin's isotropy: what it has to produce is blotches on a
// surface a few sigma across, which the smoothstep interpolation already reads as
// organic. The hash is the usual sin-fract one -- its artefacts are periodic at
// scales far outside anything on screen here.
float _hash3(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}
float _vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 w = f * f * (3.0 - 2.0 * f);        // smoothstep, so octaves have no creases
    float n000 = _hash3(i + vec3(0.0, 0.0, 0.0));
    float n100 = _hash3(i + vec3(1.0, 0.0, 0.0));
    float n010 = _hash3(i + vec3(0.0, 1.0, 0.0));
    float n110 = _hash3(i + vec3(1.0, 1.0, 0.0));
    float n001 = _hash3(i + vec3(0.0, 0.0, 1.0));
    float n101 = _hash3(i + vec3(1.0, 0.0, 1.0));
    float n011 = _hash3(i + vec3(0.0, 1.0, 1.0));
    float n111 = _hash3(i + vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, w.x), mix(n010, n110, w.x), w.y),
               mix(mix(n001, n101, w.x), mix(n011, n111, w.x), w.y), w.z);
}
float _fbm(vec3 p) {
    return 0.57 * _vnoise(p) + 0.29 * _vnoise(p * 2.0) + 0.14 * _vnoise(p * 4.0);
}

void main() {
    vec3 dir = normalize(v_rayView);          // ray from the eye (origin) out
    vec3 c = v_centerView;
    float b = dot(dir, c);
    float disc = b * b - (dot(c, c) - v_radius * v_radius);
    if (disc < 0.0) discard;                  // ray misses the sphere
    float t = b - sqrt(disc);                 // near intersection
    if (t < 0.0) discard;
    vec3 hit = dir * t;
    // Periodic-box handling (opaque, no transparency): clip every bead to the
    // box faces so a bead crossing a seam has its sliced area move continuously
    // to the wrapped ghost on the opposite face -- and beyond-box fragments are
    // discarded (no depth/color written), so they never falsely occlude what is
    // behind. The soft edge fade is done in screen space in the composite pass
    // (see its `edgeFade`), not per-bead here, so it no longer depends on which
    // world face a bead sits near.
    if (clipEnable > 0.5) {
        vec3 wp = (viewInv * vec4(hit, 1.0)).xyz;
        float dEdge = min(boxHalf.x - abs(wp.x), boxHalf.y - abs(wp.y));
        if (dEdge < 0.0) discard;             // outside the periodic cell -> gone
    }
    vec3 N = normalize(hit - c);
    // Correct per-pixel depth so overlapping beads intersect via the depth test.
    vec4 clip = proj * vec4(hit, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;
    // Banded MesoMem albedo: yellow equator (perp to the director) -> blue poles
    // (along it), same blend as the old CPU _banded_sphere_sprite. The colours
    // arrive already converted to linear light (see to_linear).
    float s = dot(N, normalize(v_dirView));   // signed cos-latitude (+ = +n pole)
    vec3 albedo;
    if (v_material > 0.5) {
        // A BODY'S OWN MATERIAL, ahead of every colouring and instead of all of
        // them (see RenderStyle.body_material for why the rod cannot wear any of
        // them). Mottled cell wall: fbm between the two body colours, plus a
        // finer, higher-contrast octave for granularity.
        //
        // NOTHING HERE MAY DEPEND ON THE SPHERE'S OWN NORMAL. A body is a row of
        // overlapping impostors, so any per-sphere term -- a rim darkening
        // toward the silhouette was the one tried -- is a stripe per sphere, and
        // the capsule comes back as the stack of coins that handing every sphere
        // an axis ACROSS the body was there to avoid (see
        // MesoMemRod.glyph_spheres). The albedo is a function of world POSITION
        // alone, and the rounding comes from the lighting pass, off the real
        // per-pixel normals, as it does for a bead.
        //
        // The noise is sampled in the OWNER's frame -- v_tint.xyz carries its
        // world position, which for this material is what that channel is for
        // (the tint itself is dead here, and adding a fourth vec3 per instance to
        // every bead in a 50k scene to serve one rod is not worth the bandwidth).
        // Anchoring it there is what keeps the texture ON the rod as it is
        // steered, instead of the rod sliding through a fixed field of blotches.
        vec3 wp = (viewInv * vec4(hit, 1.0)).xyz;
        vec3 q = (wp - v_tint.xyz) / max(bodyMottle, 1e-3);
        // Stretched past 0..1 and clipped, so the blotches have flat cores and
        // definite edges instead of everything sitting in the muddy middle of
        // the ramp -- fbm's own distribution is heaped around 0.5.
        float m = clamp(_fbm(q) * 2.1 - 0.55, 0.0, 1.0);
        albedo = mix(bodyDark, bodyLight, m);
        albedo *= 0.82 + 0.36 * _vnoise(q * 9.0);
        o_albedo = vec4(albedo * v_bright, v_fade);
        o_normal = N;
        // w = 0 marks this pixel as a BODY, and the composite's outline pass is
        // the one thing that asks. A body is a row of overlapping impostors, so
        // its surface is faintly scalloped -- the creases where consecutive
        // spheres meet -- and a depth-gradient outline finds every one of them
        // and rules the capsule like a barcode. The silhouette's depth gradient
        // is orders larger, so raising the threshold there (see
        // BODY_OUTLINE_SCALE) keeps the contour and drops the creases. Every
        // other pass reads only .xyz.
        o_viewpos = vec4(hit, 0.0);
        return;
    }
    if (colorMode > 1.5) {
        // A colour the CPU picked for this bead and nothing more -- which cluster
        // it belongs to, crossfaded (see renderer._cluster_tints). Flat for the
        // same reason the energy is: it is a fact about the whole bead.
        albedo = v_tint.rgb;
    } else if (colorMode > 0.5) {
        // One flat colour per bead, from its own potential energy. Flat on
        // purpose: the number belongs to the WHOLE bead, so shading it like a
        // band would invite reading a gradient across a sphere that has none.
        //
        // A TINTED bead may be on a scale of its own (tintEnergy): a species whose
        // number is a different quantity -- the chain's bending energy -- painted
        // in a different hue family, so it is never read against the membrane's.
        bool own = tintEnergy > 0.5 && v_tint.a > 0.5;
        vec2 rng = own ? energyRange2 : energyRange;
        float t = clamp((v_energy - rng.x) / max(rng.y - rng.x, 1e-6), 0.0, 1.0);
        float f = t * 31.0;
        int i = int(floor(f));
        albedo = own ? mix(ramp2[i], ramp2[min(i + 1, 31)], f - float(i))
                     : mix(ramp[i], ramp[min(i + 1, 31)], f - float(i));
    } else {
        float cosl = abs(s);
        float tt = clamp((cosl - (band_half - band_soft)) / (2.0 * band_soft), 0.0, 1.0);
        albedo = mix(equator_col, pole_col, tt);
        // ...and over that, whatever colour the playground gave THIS bead, by
        // its own per-bead weight (see renderer._static_tints). Only here, in the
        // banding: this is a statement about what a bead IS -- a polymer bead in
        // a membrane -- and the other two colourings are deliberate whole-scene
        // measurements that must show every bead on the one scale.
        albedo = mix(albedo, v_tint.rgb, v_tint.a);
    }
    // Over-paint the +n pole white (down to ~80% latitude) so director sense
    // reads. Kept in ALL THREE colourings: the energy tells you how bound a bead
    // is and the cluster colour tells you what it is part of, and without this cap
    // either would cost you which way it points.
    float w = smoothstep(white_min - white_soft, white_min + white_soft, s);
    albedo = mix(albedo, white_col, w);
    // The alpha channel is spare, and this is what it is for: a per-bead
    // strength the composite blends toward the background, which is how a
    // periodic image fades out with distance.
    o_albedo = vec4(albedo * v_bright, v_fade);
    o_normal = N;
    o_viewpos = vec4(hit, 1.0);   // ... a bead, not a body: see the branch above
}
"""

_FULLSCREEN_VS = """
#version 330 core
out vec2 uv;
void main() {
    // Single covering triangle; uv in [0,1].
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Shared by every screen-space pass below. `projScale` is (P[0][0], P[1][1]) of
# the projection: forward, ndc.x = P00 * x / dist, so a view-space point projects
# back to a texture coordinate with two multiplies. Assumes a SYMMETRIC frustum,
# which proj_matrix above always builds.
_COMMON = """
uniform vec2 projScale;

vec2 view_to_uv(vec3 p) {
    vec2 ndc = projScale * p.xy / max(-p.z, 1e-6);
    return ndc * 0.5 + 0.5;
}

// Cheap per-pixel hash. Used to decorrelate the sampling patterns below between
// neighbouring pixels: it trades structured error (banding, terracing) for
// unstructured error of the same size, which reads as texture instead.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
"""

_SSAO_FS = """
#version 330 core
uniform sampler2D posTex;
uniform sampler2D normalTex;
uniform int nSamples;
uniform float radius;
uniform float bias;
uniform vec3 lightDirView;   // FROM the surface TOWARD the sun, in view space
uniform float shadowOn;
uniform float shadowLen;
uniform float shadowBias;
uniform float shadowThick;
in vec2 uv;
out vec2 frag;               // (ambient occlusion, sun visibility)
""" + _COMMON + """
// ---- screen-space contact shadows ------------------------------------------
// March toward the light through the depth buffer: if anything the depth buffer
// knows about is standing in the way, this point is in shadow. Cost is per-pixel
// and independent of the bead count, the same bargain as the occlusion above --
// and it rides in this pass, at half resolution, so that the AO blur downstream
// smooths BOTH terms. That matters more here than for the AO: the march below
// is a binary hit, jittered per pixel, so on its own it comes out as dithered
// noise along every shadow edge, and blurring is what turns that noise back
// into a soft penumbra.
//
// What screen space cannot do is what the words mean: an occluder hidden behind
// something else, or off the edge of the frame, cannot cast, and how THICK an
// occluder is has to be guessed (shadowThick). That costs missing shadows, not
// ugly ones, which is the right way round.
float sun_visibility(vec3 P, vec3 N, vec3 L) {
    if (shadowOn < 0.5 || dot(N, L) <= 0.0) return 1.0;
    const int STEPS = 16;
    // Start along the NORMAL, not along the light: near a silhouette the light
    // can graze back across this very bead and the ray then reports the surface
    // shadowing itself (classic acne).
    vec3 origin = P + N * shadowBias;
    float jit = hash12(gl_FragCoord.xy);        // kills the terracing
    for (int i = 0; i < STEPS; ++i) {
        // QUADRATIC spacing: contact shadows need precision within a fraction
        // of a radius, a shadow four radii out does not.
        float f = (float(i) + jit) / float(STEPS);
        vec3 sp = origin + L * (shadowLen * f * f);
        vec2 suv = view_to_uv(sp);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) break;
        float sz = texture(posTex, suv).z;
        if (sz >= -1e-4) continue;              // background occludes nothing
        float dz = -sp.z - (-sz);
        // Occluder in front of the sample, but not so far in front that it is
        // unrelated foreground.
        if (dz > shadowBias && dz < shadowThick) return 0.0;
    }
    return 1.0;
}

void main() {
    vec3 P = texture(posTex, uv).xyz;
    if (P.z >= -1e-4) { frag = vec2(1.0); return; }              // background
    vec3 N = normalize(texture(normalTex, uv).xyz);
    frag.y = sun_visibility(P, N, normalize(lightDirView));
    if (nSamples <= 0) { frag.x = 1.0; return; }

    float rot = hash12(gl_FragCoord.xy);
    // Tangent frame about the normal, so samples land in the hemisphere ABOVE
    // the surface and never inside it.
    vec3 rv = normalize(vec3(cos(rot * 6.283), sin(rot * 6.283), 0.0));
    vec3 T = normalize(rv - N * dot(rv, N));
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    float occ = 0.0;
    for (int i = 0; i < nSamples; ++i) {
        // R2 low-discrepancy sequence: better coverage than white noise, and no
        // bit operations, so it works in GLSL 330.
        vec2 xi = fract(vec2(float(i) * 0.7548776662,
                             float(i) * 0.5698402909) + rot);
        float r = sqrt(xi.x);                        // cosine-weighted hemisphere
        float phi = 6.2831853 * xi.y;
        vec3 h = vec3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - xi.x, 0.0)));

        vec3 sp = P + TBN * h * radius;
        vec2 suv = view_to_uv(sp);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) continue;
        float sz = texture(posTex, suv).z;
        if (sz >= -1e-4) continue;                   // background: occludes nothing
        float sd = -sz, sampleDist = -sp.z;
        // Is the stored surface in FRONT of our sample point? Then the sample is
        // buried, i.e. occluded. The range check keeps a distant foreground
        // object from darkening this pixel, which would ring every silhouette.
        if (sd < sampleDist - bias)
            occ += smoothstep(0.0, 1.0, radius / max(abs(sampleDist - sd), 1e-4));
    }
    frag.x = clamp(1.0 - occ / float(nSamples), 0.0, 1.0);
}
"""

_BLUR_FS = """
#version 330 core
uniform sampler2D aoTex;
uniform sampler2D posTex;
uniform vec2 texel;
in vec2 uv;
out vec2 frag;
void main() {
    // Foreground-only 4x4 box blur of BOTH half-res terms: ambient occlusion in
    // x, sun visibility in y. Both are noisy by construction -- the AO from its
    // sample pattern, the shadow from its per-pixel jittered binary hit -- and
    // both are low-frequency signals, so a box blur is all either needs.
    //
    // The mask is what keeps it honest. The BACKGROUND's values are 1.0 (fully
    // open, fully lit), and a naive blur bleeds that into a bead's silhouette
    // pixels, un-darkening them into a bright halo around the cluster.
    // Weighting each tap by whether it is a bead fragment (view-space z < 0;
    // the background G-buffer z is cleared to 0) keeps the blur inside the
    // geometry, so no halo forms.
    vec2 sum = vec2(0.0);
    float wsum = 0.0;
    for (int x = -2; x < 2; ++x)
        for (int y = -2; y < 2; ++y) {
            vec2 o = uv + vec2(x, y) * texel;
            float fg = texture(posTex, o).z < -1e-4 ? 1.0 : 0.0;
            sum += texture(aoTex, o).rg * fg;
            wsum += fg;
        }
    frag = wsum > 0.0 ? sum / wsum : texture(aoTex, uv).rg;
}
"""

_COMPOSITE_FS = """
#version 330 core
uniform sampler2D albedoTex;
uniform sampler2D normalTex;
uniform sampler2D posTex;
uniform sampler2D aoTex;
uniform vec2 texel;
uniform vec3 lightDirView;   // FROM the surface TOWARD the sun, in view space
uniform vec3 sunColor;
uniform float sunGain;
uniform vec3 skyAmbient;
uniform float specPower;
uniform float specGain;
uniform vec3 fresnelColor;
uniform float fresnelGain;
uniform vec3 bgColor;        // linear-light background
uniform float bgGradient;    // signed brightness ramp toward the bottom of the frame
uniform float aoStrength;
uniform float curvAO;
uniform float outlineOn;
uniform float outlineStrength;
uniform float outlineThresh;
// How much higher the outline's threshold is on a body than on a bead. Big,
// because the two gradients it has to tell apart are: a silhouette (a jump of
// the whole scene's depth) and a crease between two impostors a fraction of a
// radius apart. Anywhere in this neighbourhood works; it is not a taste dial,
// which is why it is a constant here rather than a RenderStyle field.
#define BODY_OUTLINE_SCALE 12.0
uniform vec3 outlineColor;
uniform float cueOn;
uniform float cueNear;
uniform float cueFar;
uniform float cueStrength;
uniform float edgeFade;      // periodic scenes: screen-edge fade strength (0 = off)
uniform float tonemapOn;
uniform float exposure;
uniform float tonemapMix;
uniform float vignette;
uniform float invGamma;
in vec2 uv;
out vec4 frag;

vec3 aces(vec3 x) {
    // Narkowicz's cheap ACES fit: filmic highlight rolloff for one mad + div.
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    // The sky and the beads BOTH fall through to the tonemap at the bottom.
    // That matters: anything faded toward the background has to pass through
    // the same transfer curve as the background, or a "fully cued" bead lands
    // several times brighter than the void it was supposed to disappear into.
    //
    // Background test off the G-buffer view-position (z < 0 on any bead,
    // cleared to 0 on the background) rather than the depth texture -- that is
    // this framebuffer's own depth attachment, and sampling it here would be a
    // framebuffer feedback loop (undefined in GL; on a tile-based GPU like
    // Apple Silicon it surfaced as intermittent black tiles).
    // The ramp is signed: it LIFTS the bottom of a dark void into a soft glow,
    // and SINKS the bottom of a light one, where lifting it would only push the
    // background through the top of the tonemap into a flat white band.
    vec3 sky = bgColor * (1.0 + bgGradient * (1.0 - uv.y));
    vec3 P = texture(posTex, uv).xyz;
    vec3 col = sky;

    if (P.z < -1e-4) {
        float dist = -P.z;                     // distance along the view axis
        vec4 albedoSample = texture(albedoTex, uv);
        vec3 alb = albedoSample.rgb;
        float strength = albedoSample.a;
        vec3 N = normalize(texture(normalTex, uv).xyz);
        vec3 L = normalize(lightDirView);
        vec3 V = normalize(-P);                // eye is the origin in view space

        // Both terms come from the half-res pass, already blurred: x is the
        // ambient occlusion, y how much of the sun reaches this point.
        vec2 occlusion = texture(aoTex, uv).rg;
        float ao = occlusion.x;
        float shadow = occlusion.y;
        // Curvature AO, free: the rim of a curved surface is partly occluded by
        // its own body, and N.z falls to 0 exactly at the silhouette.
        if (curvAO > 0.5) ao *= 0.55 + 0.45 * smoothstep(0.0, 0.55, N.z);
        // pow(), not a multiply: this pins ao = 1 (fully open stays fully lit)
        // and bends everything below it down, so crevices deepen while open
        // surfaces keep their brightness.
        ao = pow(clamp(ao, 0.0, 1.0), aoStrength);

        float diff = max(dot(N, L), 0.0);

        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), specPower);
        float fres = pow(1.0 - max(dot(N, V), 0.0), 5.0);

        // AO darkens the AMBIENT term only -- that is what it models. Scaling
        // the sun by it too would just be a second, wrong shadow.
        col = alb * (skyAmbient * ao + sunColor * diff * shadow * sunGain);
        col += vec3(1.0) * spec * shadow * specGain;
        col = mix(col, fresnelColor, fres * fresnelGain * ao);

        // ---- outline: depth-discontinuity edge detect --------------------
        if (outlineOn > 0.5) {
            float dl = -texture(posTex, uv + vec2(-texel.x, 0.0)).z;
            float dr = -texture(posTex, uv + vec2( texel.x, 0.0)).z;
            float dd = -texture(posTex, uv + vec2(0.0, -texel.y)).z;
            float du = -texture(posTex, uv + vec2(0.0,  texel.y)).z;
            float g = max(abs(dl - dr), abs(dd - du));
            // The threshold scales with distance, or ordinary perspective
            // foreshortening would outline everything far away. And it is raised
            // on a BODY (posTex.w = 0, see the geometry shader): a body's surface
            // is a row of overlapping sphere impostors and the creases between
            // them are real depth gradients, small ones -- so this is the number
            // that separates "the edge of the object" from "where two of the
            // spheres it is built out of meet".
            float t0 = outlineThresh * dist
                     * (texture(posTex, uv).w < 0.5 ? BODY_OUTLINE_SCALE : 1.0);
            float edge = smoothstep(t0, t0 * 4.0, g);
            col = mix(col, outlineColor, clamp(edge * outlineStrength, 0.0, 1.0));
        }

        // ---- depth cue ---------------------------------------------------
        // A straight linear ramp to the background across the scene's own depth
        // span. Distinct from physical fog (exponential extinction): this is a
        // depth-perception aid from technical illustration, and in a dense bead
        // cloud it does more for legibility because the ramp spans exactly the
        // depth range that actually contains beads.
        if (cueOn > 0.5) {
            float c = clamp((dist - cueNear) / max(cueFar - cueNear, 1e-4), 0.0, 1.0);
            col = mix(col, sky, c * cueStrength);
        }

        // Periodic images: the further a copy is from the real cell, the more
        // of the background shows through it, so the tiling has no outer edge.
        // After the depth cue, because it is the same kind of statement -- this
        // is further away, let go of it -- and before the screen vignette, which
        // is about the frame rather than the scene.
        if (strength < 1.0) col = mix(sky, col, clamp(strength, 0.0, 1.0));

        // Periodic scenes: fade the beads toward the background over the outer
        // frame margin, uniformly in screen space, softening both the frame
        // edge and the periodic clip seam. Applied only to bead fragments, and
        // the box outline is drawn later, so it stays bright.
        if (edgeFade > 0.0) {
            vec2 dc = abs(uv - 0.5) * 2.0;         // 0 at center -> 1 at each edge
            col = mix(col, sky, edgeFade * smoothstep(0.55, 1.0, max(dc.x, dc.y)));
        }
    }

    if (tonemapOn > 0.5) {
        // HALF-STRENGTH ACES: the full filmic S lifts the shadows and pulls
        // saturated colour toward grey, which on a near-monochrome scene reads
        // as washed out. Blending it half-and-half with the linear colour keeps
        // the rolloff at half slope and gives the colour back. The cost is that
        // the linear half is unbounded, so a very bright specular clips instead
        // of rolling off -- hence the clamp.
        col = mix(col, aces(col * exposure), tonemapMix);
        col = pow(clamp(col, 0.0, 1.0), vec3(invGamma));
        // dot(q,q) peaks at 0.5 in the corners, so 1.1 takes them to 45%.
        vec2 q = uv - 0.5;
        col *= 1.0 - vignette * dot(q, q);
    }
    frag = vec4(col, 1.0);
}
"""

_DOF_FS = """
#version 330 core
uniform sampler2D colorTex;
uniform sampler2D posTex;
uniform vec2 texel;
uniform float focus;
uniform float range;
uniform float maxCoc;
uniform float enabled;
in vec2 uv;
out vec4 frag;

// Circle of confusion: 0 in focus, 1 maximally blurred. A real lens focuses
// exactly one distance onto the sensor; a point at any other distance projects
// as a disc, and the whole effect is the size of that disc.
float coc_of(vec3 P) {
    if (P.z >= -1e-4) return 1.0;                   // background: fully blurred
    return clamp(abs(-P.z - focus) / range, 0.0, 1.0);
}

void main() {
    vec4 c0 = texture(colorTex, uv);
    if (enabled < 0.5) { frag = c0; return; }
    float coc = coc_of(texture(posTex, uv).xyz);
    if (coc < 0.02) { frag = c0; return; }          // already sharp

    // 24 taps on a golden-angle spiral: even coverage of the disc with no
    // precomputed table and no visible sampling pattern.
    const int TAPS = 24;
    const float GA = 2.39996323;
    float radius = coc * maxCoc;

    vec3 sum = c0.rgb;
    float wsum = 1.0;
    for (int i = 1; i <= TAPS; ++i) {
        float t = float(i) / float(TAPS);
        float a = float(i) * GA;
        // sqrt(t) keeps the samples uniform over the AREA of the disc rather
        // than clustering them at the centre.
        vec2 off = vec2(cos(a), sin(a)) * sqrt(t) * radius * texel;
        // Weight each tap by its OWN circle of confusion, so a blurry
        // foreground bleeds outward instead of being clipped off by a sharp
        // background pixel. Not exact -- physically an out-of-focus foreground
        // SCATTERS over the background, and a gather filter like this one can
        // only pull inward -- but it removes the tell-tale hard edge around
        // near objects, which is the 90%-for-5% version of the fix.
        float w = max(coc_of(texture(posTex, uv + off).xyz), 0.05);
        sum += texture(colorTex, uv + off).rgb * w;
        wsum += w;
    }
    frag = vec4(sum / wsum, c0.a);
}
"""

# =============================================================================
# ANTI-ALIASING
# =============================================================================
# Every silhouette in this renderer is a hard edge. The impostor either hits its
# sphere or `discard`s, the outline is a smoothstep over a depth gradient that
# saturates within a pixel, and the periodic clip is a flat `discard` -- so all
# three come out stair-stepped, and on the sheet, where hundreds of small beads
# each contribute two or three arcs of edge, the whole image crawls.
#
# The usual answer, MSAA, is not available: a deferred renderer would have to
# keep and shade every G-buffer sample (4x the memory and the shading), and the
# impostors write gl_FragDepth, which disables early-z and makes per-sample depth
# resolve worse still. So this is FXAA (Lottes) instead -- one full-screen pass,
# no extra geometry cost, no interaction with the G-buffer at all.
#
# It works on LUMA, so it has to run after the tonemap and gamma encode (an edge
# in linear light is not where the eye sees one), and after the lines, so they
# get antialiased too. That is exactly the last thing this pipeline does.
_FXAA_FS = """
#version 330 core
uniform sampler2D srcTex;
uniform vec2 texel;
uniform float aaOn;
in vec2 uv;
out vec4 frag;

// Rec. 601 luma. The green weight dominates because the eye does; an edge that
// is invisible in luma is one FXAA is right to leave alone.
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec3 rgbM = texture(srcTex, uv).rgb;
    if (aaOn < 0.5) { frag = vec4(rgbM, 1.0); return; }

    float lM = luma(rgbM);
    float lNW = luma(texture(srcTex, uv + vec2(-1.0,  1.0) * texel).rgb);
    float lNE = luma(texture(srcTex, uv + vec2( 1.0,  1.0) * texel).rgb);
    float lSW = luma(texture(srcTex, uv + vec2(-1.0, -1.0) * texel).rgb);
    float lSE = luma(texture(srcTex, uv + vec2( 1.0, -1.0) * texel).rgb);

    // Local contrast. Below it, this is smooth shading rather than an edge, and
    // blurring it would only cost sharpness -- which is most of the screen, so
    // this early-out is also where the speed comes from.
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    float range = lMax - lMin;
    if (range < max(0.0625, lMax * 0.125)) { frag = vec4(rgbM, 1.0); return; }

    // The edge's direction, from the diagonal luma gradients: blur ALONG it,
    // never across it, which is what separates this from a plain blur.
    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)),
                     ((lNW + lSW) - (lNE + lSE)));
    // Bias the normalisation on dark pixels, where the gradients are small and
    // the direction estimate is mostly noise.
    float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpMin, -8.0, 8.0) * texel;

    // Two taps near the centre, plus two out at the ends of the span. If the
    // wider pair strays outside the neighbourhood's luma range it has walked off
    // this edge onto something else, so fall back to the narrow pair.
    vec3 rgbA = 0.5 * (texture(srcTex, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                       texture(srcTex, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(srcTex, uv - dir * 0.5).rgb +
                                     texture(srcTex, uv + dir * 0.5).rgb);
    float lB = luma(rgbB);
    frag = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}
"""

_LINE_VS = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;
in vec3 in_pos;
in vec4 in_col;
out vec4 v_col;
void main() {
    v_col = in_col;
    gl_Position = proj * view * vec4(in_pos, 1.0);
}
"""

_LINE_FS = """
#version 330 core
in vec4 v_col;
out vec4 frag;
void main() { frag = v_col; }
"""

_BLIT_FS = """
#version 330 core
uniform sampler2D srcTex;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(srcTex, uv); }
"""


class GLScene:
    """The GPU bead pipeline for one moderngl context, sized to a sim viewport."""

    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = 0
        self.height = 0
        self._inst_capacity = 0
        self._inst_vbo = None
        self._geom_vao = None
        self._line_vbo = [None, None]
        self._line_cap = [0, 0]
        self._line_vao = [None, None]
        # The style whose colour constants are currently uploaded. Only its gamma
        # affects them, so the geometry program is only re-tinted when that
        # changes rather than every frame.
        self._albedo_gamma = None

        self._build_programs()
        self._build_static_buffers()
        self.resize(width, height)

    # ---- setup --------------------------------------------------------------

    def _build_programs(self):
        c = self.ctx
        self.geom_prog = c.program(vertex_shader=_GEOM_VS, fragment_shader=_GEOM_FS)
        self.ssao_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_SSAO_FS)
        self.blur_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_BLUR_FS)
        self.comp_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_COMPOSITE_FS)
        self.dof_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_DOF_FS)
        self.fxaa_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_FXAA_FS)
        self.fxaa_prog["srcTex"].value = 0
        self.line_prog = c.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        self.blit_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_BLIT_FS)
        self.blit_prog["srcTex"].value = 0

        # Bead banding geometry (the colours themselves are per-style, below).
        self.geom_prog["band_half"].value = BEAD_BAND_HALFWIDTH
        self.geom_prog["band_soft"].value = BEAD_BAND_SOFT
        self.geom_prog["white_min"].value = BEAD_WHITE_POLE_MIN
        self.geom_prog["white_soft"].value = BEAD_WHITE_POLE_SOFT
        # Periodic bead-clipping defaults to OFF; render() sets it per frame.
        self.geom_prog["clipEnable"].value = 0.0
        self.geom_prog["boxHalf"].value = (1e9, 1e9)
        # The energy colormap never changes; only which mode is active and what
        # range it spans do (per frame, in render()).
        self.geom_prog["ramp"].write(np.array(INFERNO, dtype="f4").tobytes())
        self.geom_prog["ramp2"].write(np.array(BEND_RAMP, dtype="f4").tobytes())
        self.geom_prog["tintEnergy"].value = 0.0
        self.geom_prog["energyRange2"].value = (0.0, 1.0)
        self.geom_prog["colorMode"].value = 0.0
        self.geom_prog["energyRange"].value = (-6.0, 0.0)

        # Texture unit assignments (fixed for the life of the programs).
        self.comp_prog["albedoTex"].value = 0
        self.comp_prog["normalTex"].value = 1
        self.comp_prog["posTex"].value = 2
        self.comp_prog["aoTex"].value = 3
        self.ssao_prog["posTex"].value = 0
        self.ssao_prog["normalTex"].value = 1
        self.blur_prog["aoTex"].value = 0
        self.blur_prog["posTex"].value = 1
        self.dof_prog["colorTex"].value = 0
        self.dof_prog["posTex"].value = 1

    def _build_static_buffers(self):
        c = self.ctx
        # Quad corners for the impostor billboard (triangle strip). They
        # circumscribe the unit circle, which is what makes the silhouette
        # construction in _GEOM_VS exact.
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad_vbo = c.buffer(quad.tobytes())
        # One cached VAO per fullscreen program (vertices come from gl_VertexID,
        # so they need no buffer -- but a VAO per draw is still an allocation).
        self._fs_vaos = {
            p: c.vertex_array(p, [])
            for p in (self.ssao_prog, self.blur_prog, self.comp_prog,
                      self.dof_prog, self.fxaa_prog, self.blit_prog)
        }

    def resize(self, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        c = self.ctx
        self._release_fbos()

        # Albedo is f2 (not f1) so a per-bead brightness boost (>1) survives into
        # the composite before the final clamp.
        self.albedo_tex = c.texture((width, height), 4, dtype="f2")
        self.normal_tex = c.texture((width, height), 3, dtype="f2")
        # Full float for the view position: its z is a world-space distance, and
        # the outline pass compares NEIGHBOURING distances against a threshold of
        # a couple of thousandths of one -- which is below half-float resolution
        # out at the far side of a big box, so f2 here reads as edge noise.
        self.pos_tex = c.texture((width, height), 4, dtype="f4")
        self.depth_tex = c.depth_texture((width, height))
        self.depth_tex.compare_func = ""          # sample raw depth, not a compare
        for t in (self.albedo_tex, self.normal_tex, self.pos_tex):
            t.filter = (c.NEAREST, c.NEAREST)
        self.gbuffer = c.framebuffer(
            color_attachments=[self.albedo_tex, self.normal_tex, self.pos_tex],
            depth_attachment=self.depth_tex,
        )

        # Ambient occlusion and sun visibility, two channels of one texture at
        # HALF resolution: both are broad, low-frequency terms that get blurred
        # anyway, so a quarter of the pixels buys a quarter of the cost for no
        # visible difference. LINEAR, because the composite then samples them at
        # full res and nearest would show the half-res grid.
        aw, ah = max(1, width // 2), max(1, height // 2)
        self.ao_size = (aw, ah)
        self.ao_tex = c.texture((aw, ah), 2, dtype="f2")
        self.ao_blur_tex = c.texture((aw, ah), 2, dtype="f2")
        for t in (self.ao_tex, self.ao_blur_tex):
            t.filter = (c.LINEAR, c.LINEAR)
        self.ao_fbo = c.framebuffer(color_attachments=[self.ao_tex])
        self.ao_blur_fbo = c.framebuffer(color_attachments=[self.ao_blur_tex])

        # The composite renders HERE rather than to the final texture, so the
        # depth-of-field pass has something to read. LINEAR: DoF samples it
        # off-grid.
        self.lit_tex = c.texture((width, height), 4, dtype="f2")
        self.lit_tex.filter = (c.LINEAR, c.LINEAR)
        self.lit_fbo = c.framebuffer(color_attachments=[self.lit_tex])

        # The finished picture, before antialiasing: depth of field lands here and
        # the lines are drawn into it, sharing the geometry depth buffer so they
        # are occluded by the beads. LINEAR because FXAA reads it off-grid.
        self.shaded_tex = c.texture((width, height), 4, dtype="f1")
        self.shaded_tex.filter = (c.LINEAR, c.LINEAR)
        self.shaded_fbo = c.framebuffer(
            color_attachments=[self.shaded_tex], depth_attachment=self.depth_tex
        )
        # ...and after: what gets blitted to the window and read back for
        # snapshots, so both see the same image.
        self.final_tex = c.texture((width, height), 4, dtype="f1")
        self.final_fbo = c.framebuffer(color_attachments=[self.final_tex])

        self.blur_prog["texel"].value = (1.0 / aw, 1.0 / ah)
        self.comp_prog["texel"].value = (1.0 / width, 1.0 / height)
        self.dof_prog["texel"].value = (1.0 / width, 1.0 / height)
        self.fxaa_prog["texel"].value = (1.0 / width, 1.0 / height)

    def _release_fbos(self):
        for attr in ("gbuffer", "ao_fbo", "ao_blur_fbo", "lit_fbo", "shaded_fbo",
                     "final_fbo", "albedo_tex", "normal_tex", "pos_tex",
                     "depth_tex", "ao_tex", "ao_blur_tex", "lit_tex",
                     "shaded_tex", "final_tex"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
                setattr(self, attr, None)

    # ---- style --------------------------------------------------------------

    def _apply_style(self, style, bead_radius, light_dir_view, depth_range,
                     edge_fade, focal_px):
        """Push one RenderStyle into the shader uniforms. Called per frame, so a
        slider or a per-system override takes effect immediately; the cost is a
        few dozen uniform writes, which is nothing beside one full-screen pass."""
        gamma = style.display_gamma
        if gamma != self._albedo_gamma:
            self.geom_prog["equator_col"].value = to_linear(BEAD_EQUATOR_COLOR, gamma)
            self.geom_prog["pole_col"].value = to_linear(BEAD_POLE_COLOR, gamma)
            self.geom_prog["white_col"].value = to_linear(BEAD_WHITE_POLE_COLOR, gamma)
            self._albedo_gamma = gamma
        # The body material's two colours. Per style rather than cached with the
        # bead bands above, because a playground declares them (the bands are
        # fixed by the model) -- and a uniform write is nothing.
        self.geom_prog["bodyLight"].value = to_linear(style.body_color_light, gamma)
        self.geom_prog["bodyDark"].value = to_linear(style.body_color_dark, gamma)
        self.geom_prog["bodyMottle"].value = style.body_mottle_r * max(
            float(bead_radius), 1e-6)

        r = max(float(bead_radius), 1e-6)
        light = tuple(float(v) for v in light_dir_view)
        # The half-res pass owns both occlusion terms, so the shadow march's
        # scales and the light direction go here, not to the composite.
        a = self.ssao_prog
        a["nSamples"].value = int(style.ao_samples)
        a["radius"].value = style.ao_radius_r * r
        a["bias"].value = style.ao_bias_r * r
        a["lightDirView"].value = light
        a["shadowOn"].value = 1.0 if style.shadows else 0.0
        a["shadowLen"].value = style.shadow_len_r * r
        a["shadowBias"].value = style.shadow_bias_r * r
        a["shadowThick"].value = style.shadow_thick_r * r

        p = self.comp_prog
        p["lightDirView"].value = light
        p["sunColor"].value = tuple(style.sun_color)
        p["sunGain"].value = style.sun_gain
        p["skyAmbient"].value = tuple(style.sky_ambient)
        p["specPower"].value = style.spec_power
        p["specGain"].value = style.spec_gain
        p["fresnelColor"].value = tuple(style.fresnel_color)
        p["fresnelGain"].value = style.fresnel_gain
        p["bgColor"].value = to_linear(style.background, gamma)
        p["bgGradient"].value = style.background_gradient
        p["aoStrength"].value = style.ao_strength
        p["curvAO"].value = 1.0 if style.curvature_ao else 0.0
        p["outlineOn"].value = 1.0 if style.outline else 0.0
        p["outlineStrength"].value = style.outline_strength
        p["outlineThresh"].value = style.outline_threshold(focal_px)
        p["outlineColor"].value = tuple(style.outline_color)
        cue_near, cue_far = style.cue_range(*depth_range)
        p["cueOn"].value = 1.0 if style.depth_cue else 0.0
        p["cueNear"].value = cue_near
        p["cueFar"].value = cue_far
        p["cueStrength"].value = style.cue_strength
        p["edgeFade"].value = float(edge_fade)
        p["tonemapOn"].value = 1.0 if style.tonemap else 0.0
        p["exposure"].value = style.tonemap_exposure
        p["tonemapMix"].value = style.tonemap_mix
        p["vignette"].value = style.vignette
        p["invGamma"].value = 1.0 / gamma

        focus, rng = style.focus_range(*depth_range)
        d = self.dof_prog
        d["enabled"].value = 1.0 if (style.dof and style.dof_bokeh_px > 0.0) else 0.0
        d["focus"].value = focus
        d["range"].value = rng
        d["maxCoc"].value = (style.dof_bokeh_px * self.height / DOF_REFERENCE_HEIGHT)

        self.fxaa_prog["aaOn"].value = 1.0 if style.antialias else 0.0

    # ---- per-frame instance / line uploads ----------------------------------

    def _upload_instances(self, centers, radii, directors, brights, energies,
                          tints, fades, materials):
        """Upload all beads (opaque) into the geometry VAO. Layout is 15 floats
        per instance: center(3), radius(1), director(3), brightness(1),
        energy(1), tint(4), fade(1), material(1).

        The energy and the tint are the two colourings that are not derived from
        the bead's own geometry, and only one of them is live at a time (see
        `colorMode`), so four of those fifteen floats are always dead. Kept as
        separate channels anyway: the energy is a NUMBER the shader ramps, which
        is what lets the ramp be retuned without touching the CPU, while the tint
        is already a colour, because what it encodes -- which cluster, crossfaded
        -- is not a number the shader could map.

        `material` is one float rather than a colour because it selects a whole
        shading BRANCH (a body's own material -- see RenderStyle.body_material),
        and that branch has no use for the tint, so the tint's three colour
        channels carry its noise anchor instead. One float per bead for a feature
        one playground uses is a fair price; four more would not be."""
        n = len(centers)
        if n == 0:
            return 0
        data = np.empty((n, 15), dtype="f4")
        data[:, 0:3] = centers
        data[:, 3] = radii
        data[:, 4:7] = directors
        data[:, 7] = brights
        data[:, 8] = energies
        data[:, 9:13] = tints
        data[:, 13] = fades
        data[:, 14] = materials
        raw = data.tobytes()
        if n > self._inst_capacity:
            if self._inst_vbo is not None:
                self._inst_vbo.release()
            if self._geom_vao is not None:
                self._geom_vao.release()
            self._inst_vbo = self.ctx.buffer(reserve=max(1, n) * 15 * 4, dynamic=True)
            self._inst_capacity = n
            self._geom_vao = self.ctx.vertex_array(
                self.geom_prog,
                [(self._quad_vbo, "2f", "in_corner"),
                 (self._inst_vbo, "3f 1f 3f 1f 1f 4f 1f 1f /i", "in_center", "in_radius",
                  "in_dir", "in_bright", "in_energy", "in_tint", "in_fade",
                  "in_material")],
            )
        self._inst_vbo.write(raw)
        return n

    def _upload_lines(self, slot, verts, colors):
        """verts: (M,3) segment endpoints (pairs), colors: (M,4) rgba in [0,1].

        Two slots, because the line pass is drawn twice: once depth-tested
        against the beads, and once over the top of them (see render())."""
        m = len(verts)
        data = np.empty((m, 7), dtype="f4")
        data[:, 0:3] = verts
        data[:, 3:7] = colors
        if m > self._line_cap[slot]:
            for obj in (self._line_vbo[slot], self._line_vao[slot]):
                if obj is not None:
                    obj.release()
            self._line_vbo[slot] = self.ctx.buffer(reserve=max(1, m) * 7 * 4,
                                                   dynamic=True)
            self._line_cap[slot] = m
            self._line_vao[slot] = self.ctx.vertex_array(
                self.line_prog,
                [(self._line_vbo[slot], "3f 4f", "in_pos", "in_col")],
            )
        self._line_vbo[slot].write(data.tobytes())
        return m

    # ---- render -------------------------------------------------------------

    def render(self, view, proj, centers, radii, directors, depth_range,
               line_verts=None, line_colors=None, brights=None,
               box_half=None, edge_fade=0.0, style=DEFAULT_STYLE,
               bead_radius=None, focal_px=None, light_dir_world=None,
               energies=None, tints=None, fades=None, materials=None,
               overlay_verts=None, overlay_cols=None, color_mode=None):
        """Render the beads (+ optional depth-occluded lines) into self.final_fbo.

        view/proj are row-major 4x4 numpy matrices (see view_matrix/proj_matrix).
        `depth_range` is (nearest, farthest) bead distance along the view axis;
        the depth cue and the focus plane are placed as fractions of it, so they
        follow the scene rather than needing absolute world numbers.
        `brights` (albedo multiplier) is an optional per-bead array; `bead_radius`
        the radius the AO/shadow reaches scale off (default: the median radius);
        `focal_px` the camera's focal length in pixels, which is how big a bead
        lands on screen and so how wide its outline should be (default: derived
        from the projection).
        For a periodic scene, pass `box_half=(hx, hy)` (world half-extents, box
        centered at the origin): beads are then clipped to the box faces so
        wrapped ghosts compose correctly (opaque, no transparency), and
        `edge_fade` (0..1) softens the resulting seam at the frame edge.
        `light_dir_world` overrides the style's sun direction (world space).
        `energies` (per bead), when given, switches the beads from the director
        banding to the energy colormap over `style.energy_range`; `tints` (per
        bead, linear-light rgb plus a mix weight) is the flat colour the cluster
        colouring paints with, and, at a per-bead weight under the banding, the
        colour a playground gives its own species. Which of the three is showing
        is `color_mode` (0 bands, 1 energy, 2 tint); None derives it from which
        arrays arrived, which is the older behaviour and is what a caller that
        only ever paints clusters wants. `fades` (per bead, 1 = full strength) blends a bead toward the
        background, which is how periodic image copies are made to trail off.
        `materials` (per bead, 0 = an ordinary bead) switches an instance to a
        body's own material instead of any colouring -- see
        RenderStyle.body_material, and note that such an instance's `tints` rgb is
        read as its noise anchor rather than as a colour.
        `overlay_verts/cols` are lines drawn over everything, depth test off.
        """
        c = self.ctx
        vb, pb = _gl(view), _gl(proj)
        n = len(centers)
        centers = np.asarray(centers, "f4")
        radii = np.asarray(radii, "f4")
        directors = np.asarray(directors, "f4")
        brights = np.ones(n, "f4") if brights is None else np.asarray(brights, "f4")
        # The colouring is whichever channel the caller supplied: the director
        # banding needs neither, and is what is left when neither arrives.
        self.geom_prog["colorMode"].value = float(
            color_mode if color_mode is not None
            else (2 if tints is not None else (0 if energies is None else 1)))
        self.geom_prog["energyRange"].value = tuple(style.energy_range)
        own = style.tint_energy_range
        self.geom_prog["tintEnergy"].value = 0.0 if own is None else 1.0
        if own is not None:
            self.geom_prog["energyRange2"].value = tuple(own)
        energies = np.zeros(n, "f4") if energies is None else np.asarray(energies, "f4")
        tints = np.zeros((n, 4), "f4") if tints is None else np.asarray(tints, "f4")
        fades = np.ones(n, "f4") if fades is None else np.asarray(fades, "f4")
        materials = (np.zeros(n, "f4") if materials is None
                     else np.asarray(materials, "f4"))
        if bead_radius is None:
            bead_radius = float(np.median(radii)) if n else 1.0

        # The sun is fixed in the WORLD, so an orbiting camera moves through a lit
        # scene instead of dragging the highlight around with it. Rotating it into
        # view space here (the view matrix's rotation block; a direction has no
        # translation) keeps the composite free of the view matrix.
        sun_world = _normalize(np.asarray(
            style.sun_dir if light_dir_world is None else light_dir_world,
            dtype=np.float64))
        light_dir_view = np.asarray(view, dtype=np.float64)[:3, :3] @ sun_world

        # Only the half-res pass projects sample points back to the screen (for
        # both the occlusion hemisphere and the shadow march); the composite is
        # a pure per-pixel shade and needs no projection at all.
        proj_scale = (float(proj[0, 0]), float(proj[1, 1]))
        self.ssao_prog["projScale"].value = proj_scale
        if focal_px is None:
            focal_px = proj_scale[1] * self.height / 2.0
        self._apply_style(style, bead_radius, light_dir_view, depth_range,
                          edge_fade, focal_px)

        n_op = self._upload_instances(centers, radii, directors, brights,
                                      energies, tints, fades, materials)

        # --- geometry pass -> G-buffer (all beads, opaque) ---
        self.geom_prog["view"].write(vb)
        self.geom_prog["proj"].write(pb)
        # view -> world, for the two things in the geometry shader that need a
        # bead's WORLD position: clipping to the periodic cell's faces, and
        # anchoring a body material's noise (see RenderStyle.body_material). One
        # 4x4 inverse a frame, so it is written unconditionally rather than only
        # for the periodic scenes -- the alternative is a stale matrix reaching a
        # non-periodic scene that has a body in it, which is exactly the rod.
        self.geom_prog["viewInv"].write(_gl(np.linalg.inv(view)))
        # Periodic clipping of beads to the box faces (opaque). Off for non-periodic.
        if box_half is not None:
            self.geom_prog["boxHalf"].value = (float(box_half[0]), float(box_half[1]))
            self.geom_prog["clipEnable"].value = 1.0
        else:
            self.geom_prog["clipEnable"].value = 0.0
        self.gbuffer.use()
        c.viewport = (0, 0, self.width, self.height)
        # Clear albedo/normal to 0 and pos.z to 0 (background), depth to 1.
        self.gbuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        c.enable(c.DEPTH_TEST)
        c.depth_func = "<"
        c.disable(c.BLEND)
        if n_op:
            self._geom_vao.render(mode=c.TRIANGLE_STRIP, instances=n_op)
        c.disable(c.DEPTH_TEST)

        # --- occlusion (half res) -> ao_fbo, then the foreground-masked blur ---
        # One pass computes both ambient occlusion and sun visibility, and one
        # blur smooths both. Skipped entirely when neither is switched on, in
        # which case the cleared (1, 1) reads as "fully open, fully lit".
        self.pos_tex.use(0)
        self.normal_tex.use(1)
        self.ao_fbo.use()
        c.viewport = (0, 0, *self.ao_size)
        self.ao_fbo.clear(1.0, 1.0, 1.0, 1.0)
        ao_tex = self.ao_tex
        if style.ao_samples > 0 or style.shadows:
            self._fs_vaos[self.ssao_prog].render(mode=c.TRIANGLES, vertices=3)
            self.ao_tex.use(0)
            self.pos_tex.use(1)
            self.ao_blur_fbo.use()
            self._fs_vaos[self.blur_prog].render(mode=c.TRIANGLES, vertices=3)
            ao_tex = self.ao_blur_tex

        # --- composite (lighting + AO + shadows + outline + cue + tonemap) ---
        self.albedo_tex.use(0)
        self.normal_tex.use(1)
        self.pos_tex.use(2)
        ao_tex.use(3)
        self.lit_fbo.use()
        c.viewport = (0, 0, self.width, self.height)
        self._fs_vaos[self.comp_prog].render(mode=c.TRIANGLES, vertices=3)

        # --- depth of field -> shaded_fbo ---
        self.lit_tex.use(0)
        self.pos_tex.use(1)
        self.shaded_fbo.use()
        self._fs_vaos[self.dof_prog].render(mode=c.TRIANGLES, vertices=3)

        # --- depth-occluded lines (bonds + net + box) into the same FBO ---
        # After the blur and the tonemap on purpose: these are drawing, not
        # photography -- a defocused box outline reads as a rendering bug, and
        # their colours are already the display-space values theme.py declares.
        if ((line_verts is not None and len(line_verts) >= 2)
                or (overlay_verts is not None and len(overlay_verts) >= 2)):
            self.line_prog["view"].write(vb)
            self.line_prog["proj"].write(pb)
            self.shaded_fbo.use()
            c.enable(c.BLEND)
            if line_verts is not None and len(line_verts) >= 2:
                m = self._upload_lines(0, np.asarray(line_verts, "f4"),
                                       np.asarray(line_colors, "f4"))
                c.enable(c.DEPTH_TEST)
                c.depth_func = "<="
                c.depth_mask = False           # test against beads, don't write
                self._line_vao[0].render(mode=c.LINES, vertices=m)
                c.depth_mask = True
                c.disable(c.DEPTH_TEST)
            if overlay_verts is not None and len(overlay_verts) >= 2:
                # No depth test at all: these are the lines that have to stay
                # readable whatever is in front of them -- the cell outline, when
                # the cell itself is buried under its own periodic images.
                m = self._upload_lines(1, np.asarray(overlay_verts, "f4"),
                                       np.asarray(overlay_cols, "f4"))
                self._line_vao[1].render(mode=c.LINES, vertices=m)
            c.disable(c.BLEND)

        # --- antialias the finished picture -> final_fbo ---
        # Last, and on the gamma-encoded image: FXAA looks for edges the way the
        # eye does, and the lines above are edges too.
        self.shaded_tex.use(0)
        self.final_fbo.use()
        self._fs_vaos[self.fxaa_prog].render(mode=c.TRIANGLES, vertices=3)

    # ---- output -------------------------------------------------------------

    def blit_to_viewport(self, x, y, w, h):
        """Draw the rendered sim view into the default framebuffer's sim viewport
        rectangle (x, y, w, h) in GL (origin bottom-left) coordinates. Both this
        texture and the target are GL, so no vertical flip is needed."""
        c = self.ctx
        c.screen.use()
        c.viewport = (int(x), int(y), int(w), int(h))
        c.disable(c.DEPTH_TEST)
        c.disable(c.BLEND)
        self.final_tex.use(0)
        self._fs_vaos[self.blit_prog].render(mode=c.TRIANGLES, vertices=3)

    def read_rgb(self):
        """Read the final color buffer back as an (H, W, 3) uint8 array (row 0 =
        top), for snapshots and the benchmark."""
        data = self.final_fbo.read(components=3, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return img[::-1]                       # GL origin is bottom-left

    def release(self):
        self._release_fbos()
        for obj in (self._inst_vbo, self._geom_vao, self._quad_vbo,
                    *self._line_vbo, *self._line_vao,
                    *getattr(self, "_fs_vaos", {}).values()):
            if obj is not None:
                obj.release()
