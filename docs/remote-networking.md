# The wire, in code

How bytes actually get from a LAMMPS on a Snellius A100 to the window on your
laptop, and how a slider gets back. This is the *code-level* companion to
[remote-gpu.md](remote-gpu.md) (which is the design and the reasoning) — every
section here points at real lines and every snippet either is the shipped code or
runs against it.

Written for someone who is comfortable in Python and NumPy but does not spend
their time in sockets. Nothing below needs prior network knowledge; §1 supplies
the five facts everything else rests on.

Files, in the order the bytes pass through them:

| file | what it owns |
|---|---|
| [`remote/session.py`](../lammps_live/remote/session.py) | SSH + Slurm: gets a GPU, ships the code, opens the tunnel |
| [`remote/server.py`](../lammps_live/remote/server.py) | the far end: listens, integrates, sends frames |
| [`remote/protocol.py`](../lammps_live/remote/protocol.py) | the codec and the framing — the only file that touches bytes |
| [`remote/client.py`](../lammps_live/remote/client.py) | this end: reads frames, decodes, hands them to the app |

---

## 1. Five facts about TCP, and that is all you need

**A socket is a byte pipe between two programs.** One side `listen`s on a port
(a number: ours is 5723), the other `connect`s to it. After that either side can
`send` and the other `recv`s, in order, until someone closes.

**TCP does not lose, duplicate or reorder bytes.** That is why there are no
sequence numbers, checksums or retransmits anywhere in this codebase. If a byte
was sent, it arrives, in order — or the connection breaks and you find out. There
is no third outcome. (There *is* a `seq` on each frame, but it is for the client's
own bookkeeping — telling an old run's frames from a new one's after a Reset — not
for reliability.)

**TCP has no concept of a message.** This is the one that bites people. You send
100,000 bytes; the other end may see 1,400 then 8,192 then 90,408. It may also
see *two* of your sends glued into one read. So every protocol on top of TCP has
to say where its messages end — length prefixes here (§2).

**`recv(n)` may return fewer than `n` bytes.** The direct consequence of the
above, and the classic socket bug. It shows up only under load, which is exactly
when the frames are biggest. Handled once, in `_recv_exactly`
([protocol.py:74](../lammps_live/remote/protocol.py#L74)).

**`send` can block.** If the receiver is not reading fast enough, the kernel
buffers fill and `sendall` simply waits. This is *flow control*, and here it is a
feature: a client that falls behind slows the server down rather than letting a
backlog build. It is also why the server reads control messages on a separate
thread (§6) — a slider must not queue behind a frame that is blocked.

Two other terms you will meet below:

- **`127.0.0.1` / `localhost` — "this machine."** A server bound to 127.0.0.1 is
  reachable only from processes on the same computer. That is the whole point of
  the tunnel: it makes the remote server *look* local.
- **`0.0.0.0` — "every interface."** Reachable from the network. Used only in the
  one-hop fallback tunnel mode; see §9.

---

## 2. The envelope: length-prefixed messages

Every message in both directions has the same shape:

```
+----------------+-------------------+---------------+------------------+
| uint32 json_len| uint32 payload_len|   JSON header |   raw array bytes|
|   (4 bytes)    |    (4 bytes)      |   (json_len)  |   (payload_len)  |
+----------------+-------------------+---------------+------------------+
        <----- fixed 8-byte prefix ----->
```

Both lengths are up front so the reader can do exactly **two reads of known size**
and never has to scan for a delimiter inside binary data (a delimiter would be a
disaster here — arbitrary float bytes contain every byte value, including whatever
you picked as the terminator).

The whole sender is four lines
([protocol.py:68](../lammps_live/remote/protocol.py#L68)):

```python
_HEADER = struct.Struct("!II")     # ! = network byte order, I = uint32

def pack(header, payload=b""):
    """One message: a JSON header, optionally followed by raw array bytes."""
    js = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(js), len(payload)) + js + bytes(payload)
```

`"!II"` is `struct` shorthand: `!` means big-endian ("network byte order", the
convention for anything on a wire), `I` means unsigned 32-bit. So `_HEADER.pack(211,
100000)` gives exactly 8 bytes. *(Note the asymmetry: the lengths are big-endian,
but the NumPy arrays in the payload go out in the sender's native byte order — the
manifest carries the dtype string `'<u2'`, which encodes the endianness, so the
decoder gets it right regardless. Both ends being x86 makes this moot in practice,
but the manifest means it would still work if one were not.)*

And the receiver ([protocol.py:97](../lammps_live/remote/protocol.py#L97)):

```python
def recv_message(sock):
    """(header, payload) for the next message, or (None, None) at end of stream."""
    head = _recv_exactly(sock, _HEADER.size)
    if head is None:
        return None, None                       # clean hangup
    js_len, payload_len = _HEADER.unpack(bytes(head))
    if js_len > 1 << 20 or payload_len > 1 << 30:
        raise ProtocolError(f"implausible message lengths {js_len}/{payload_len}")
    js = _recv_exactly(sock, js_len)
    ...
    payload = _recv_exactly(sock, payload_len) if payload_len else b""
    header = json.loads(bytes(js).decode("utf-8"))
    return header, bytes(payload)
```

Two things worth pausing on:

**The sanity check on the lengths is not paranoia theatre.** The next thing the
code does is allocate a buffer of that size. Without the check, a corrupt or
hostile 4-byte prefix reading `0xFFFFFFFF` makes the process try to allocate 4 GB.
A frame is ~65 kB at 10k beads, so 1 MB of JSON and 1 GB of payload are both
absurd; anything
past that is a broken stream, not a big frame.

**`None` versus an exception.** `recv` returning zero bytes means *the other side
closed cleanly*. If that happens between messages it is a normal shutdown → return
`(None, None)`. If it happens halfway through a message the stream was truncated →
`ProtocolError`. The distinction propagates all the way up to what the HUD says.

### `_recv_exactly`, the loop that makes it work

```python
def _recv_exactly(sock, n):
    if n == 0:
        return b""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:], n - got)
        if not chunk:
            if got == 0:
                return None                     # clean end of stream
            raise ProtocolError(f"stream ended {n - got} bytes into a {n}-byte read")
        got += chunk
    return buf
```

The naive version, `b"".join(chunks)`, copies the whole 65 kB frame once per
chunk received — with an MTU of ~1500 bytes that is ~70 copies of a growing buffer
per frame, 60 times a second. `recv_into` writes straight into a pre-allocated
`bytearray`, and the `memoryview` slice is a *window* onto it rather than a copy,
so the data is written exactly once at its final address.

---

## 3. A frame, byte by byte

Run this against the real codec (`./venv/bin/python`, no LAMMPS needed):

```python
import numpy as np, struct
from lammps_live.remote import protocol
from lammps_live.playground.state import Box, FrameState

L, n = 37.41, 10_000
box = Box((0., 0., 0.), (L, L, L), (True, True, True))
rng = np.random.default_rng(0)
pos = rng.uniform(0, L, size=(n, 3))
d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
state = FrameState(positions=pos, directors=d, types=None,
                   ids=np.arange(1, n + 1), box=box)

manifest, payload = protocol.encode_frame(state, box, codec="q12")
print(manifest)          # [['pos', '|u1', [45000], [10000, 3]],
                         #  ['dir', '|u1', [10000, 2]]]
print(len(payload))      # 65000  -> exactly 6.5 bytes/bead

msg = protocol.pack({"t": "frame", "seq": 1, "n": n, "codec": "q12",
                     "arrays": manifest, "sim_time": 12.5, "dt": 0.2,
                     "thermo": [0.2, 0., 0., 0., 0.], "playing": True,
                     "unstable": None, "wall": 1.7e9}, payload)
print(len(msg), struct.unpack("!II", msg[:8]))   # 65227 (219, 65000)
```

So one frame on the wire is **65,227 bytes**: 8 bytes of prefix, 219 bytes of
JSON, 65,000 bytes of arrays. At 60 fps that is **3.9 MB/s, 31 Mbit/s** — the
JSON header is 0.3% overhead and not worth optimising.

Note the `pos` entry: **four** fields, not three. Its own shape says 45,000 bytes,
which is what the payload holds; the fourth says those bytes decode back to
(10000, 3). §4 says why it has to.

The header is deliberately fat with *metadata* and thin on data. It carries the
sim time, the thermo tuple, whether the far side is playing, whether it thinks it
is unstable — all things the HUD and the plot panels need, all a handful of bytes.
See `_send_frame` ([server.py:437](../lammps_live/remote/server.py#L437)).

### The manifest is the schema

```python
manifest = [[name, arr.dtype.str, list(arr.shape)] + ([list(logical)] if logical
                                                      else [])
            for name, arr, logical in arrays]
payload  = b"".join(np.ascontiguousarray(arr).tobytes()
                    for _name, arr, _logical in arrays)
```

The arrays are simply concatenated, and the header says what they were:
`[['pos', '|u1', [45000], [10000, 3]], ['dir', '|u1', [10000, 2]]]`. The decoder
walks that list with a running offset and never needs built-in knowledge of which
fields this particular frame chose to carry:

```python
offset = 0
for entry in manifest:
    if not 3 <= len(entry) <= 4:
        raise ProtocolError(f"manifest entry has {len(entry)} fields, want 3 or 4")
    name, dtype_str, shape = entry[0], entry[1], entry[2]
    dtype = np.dtype(dtype_str)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if offset + nbytes > len(payload):
        raise ProtocolError(f"payload too short for array {name!r}")
    raw[name] = np.frombuffer(payload, dtype=dtype, count=int(np.prod(shape)),
                              offset=offset).reshape(shape)
    logical[name] = entry[3] if len(entry) == 4 else None
    offset += nbytes
```

The optional fourth field is the logical shape a *packed* array decodes back to;
§4 says why one is needed at all. Everything else about the walk is unchanged, and
that is the point — the schema grew a field rather than the decoder growing a
special case.

`np.frombuffer` is the payoff: it does **not copy**. It hands back an array that
*views* the received bytes directly. Decoding 65 kB of positions costs one pass
of arithmetic (the dequantise and, for q12, a handful of shifts), not a parse.

This is also what makes the energies optional — when the bead colouring is off,
`'pe'` is simply absent from the manifest and those 10 kB/frame are never sent.
Nothing else changes.

---

## 4. Where the bytes were saved: quantisation

Positions and directors as plain `float32` is 24 B/bead. The `q12` codec gets it
to 6.5. `q16`, which spends 16 bits everywhere and costs 10, is still selectable
— `--codec q16` on the server, or `codec=` on the `RemoteTarget`.

Both bit counts were chosen against **the picture**, not against a round number,
and the useful discipline is to convert every candidate into pixels before
arguing about it.

### Positions: 3 × 12 bits across the cell (4.5 bytes)

Map the cell's extent onto the codes 0…`top`
([protocol.py:150](../lammps_live/remote/protocol.py#L150)):

```python
def quantise(values, lo, hi, dtype, top=None):
    top = int(np.iinfo(dtype).max if top is None else top)
    span = float(hi) - float(lo)
    if span <= 0.0:
        return np.zeros(np.shape(values), dtype=dtype)
    scaled = (np.asarray(values, dtype=np.float64) - float(lo)) * (top / span)
    return np.clip(np.rint(scaled), 0, top).astype(dtype)

def dequantise(codes, lo, hi, top=None):
    top = int(np.iinfo(codes.dtype).max if top is None else top)
    span = (float(hi) - float(lo)) / top
    return (codes.astype(np.float32) * np.float32(span)) + np.float32(lo)
```

`top` is separate from the dtype because a 12-bit field has no container of its
own: it is produced as `uint16` here and only becomes 12 bits wide in the packer
below. Nothing infers it — both ends read `POS_TOP`.

On the 50k playground's 64σ cell, one quantisation step is:

| bits/axis | step | windowed (820×900) | fullscreen | B/bead |
|---|---|---|---|---|
| 16 | 0.0010 σ | 0.01 px | 0.02 px | 6.00 |
| **12** | **0.016 σ** | **0.20 px** | **0.33 px** | **4.50** |
| 10 | 0.065 σ | 0.82 px | 1.31 px | 3.75 |
| 8 | 0.259 σ | 3.28 px | 5.24 px | 3.00 |

10 bits is the tempting one — three of them fit a `uint32` with no bit fiddling at
all — and it is the one to refuse. A step of 0.8 px windowed and 1.3 fullscreen is
above the noise floor of the eye for a bead that is *briefly almost still*, which
happens constantly in a coarsening membrane: the bead stops drifting and starts
visibly snapping between grid points. 12 bits puts the step a fifth of a pixel
down and costs three quarters of a byte more.

Two decisions inside `quantise` that are easy to get wrong:

- **The range is the cell plus 1σ of padding** (`QUANT_PAD`). LAMMPS remaps atoms
  into the periodic cell on a neighbour rebuild, not every step, so a coordinate
  can legitimately sit slightly outside its box. Padding costs 3% of the
  resolution and avoids beads visibly piling up on the wall.
- **`np.clip`, not a wrap.** Out-of-range input is clamped. A wraparound would put
  a bead on the *opposite side of the cell* — the one error that could never be
  mistaken for noise.

### Packing 12-bit fields: two codes, three bytes

Three 12-bit numbers do not fit a machine word and 4.5 bytes is not an address,
so the packing runs over the **flat stream of codes**, not per bead: any two
adjacent codes share three bytes ([protocol.py:184](../lammps_live/remote/protocol.py#L184)).

```
byte 0        byte 1                    byte 2
a[7:0]        b[3:0] | a[11:8]          b[11:4]
```

```python
def pack12(codes):
    v = np.ascontiguousarray(codes, dtype=np.uint16).ravel()
    if len(v) % 2:
        v = np.append(v, np.uint16(0))
    a, b = v[0::2].astype(np.uint16), v[1::2].astype(np.uint16)
    out = np.empty((len(a), 3), dtype=np.uint8)
    out[:, 0] = (a & 0xFF).astype(np.uint8)
    out[:, 1] = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).astype(np.uint8)
    out[:, 2] = ((b >> 4) & 0xFF).astype(np.uint8)
    return out.reshape(-1)
```

Three details, each of which is a decision:

- **The low nibble of the middle byte is `a`'s top, not `b`'s bottom.** Purely so
  the little-endian reading of the first two bytes *is* `a` — which is what makes
  a hexdump legible on the day this goes wrong.
- **Codes are laid out bead-major** (x, y, z, x, y, z…) rather than by axis. The
  pair that shares three bytes is then almost always two axes of *one* bead, so a
  truncated payload loses whole beads off the end instead of one coordinate of
  every bead in the second half of the frame.
- **An odd stream carries one padding code**, dropped on the way back out. This
  is the reason `bytes_per_bead("q12")` returns `6.5` rather than an integer, and
  the reason the round-trip test asserts the payload size to within one byte: the
  odd half really is shared with the next bead rather than rounded away.

### …which is why the manifest grew a fourth field

A packed buffer's own length is **one bead ambiguous** — 3M bytes is either 2M
codes or 2M−1 — so the array's shape no longer says how many beads it holds. The
alternative was to teach the decoder to read `n` off the frame header, and that
would have cost the property the manifest exists for: that it is the *whole*
schema, and the decoder needs no built-in knowledge of the frame. So a q12 `pos`
entry carries the logical shape as an optional fourth field, and every entry that
does not need one still has exactly the three it always had:

```python
[['pos', '|u1', [45000], [10000, 3]], ['dir', '|u1', [10000, 2]]]
```

A frame that omits it is refused rather than guessed at
([protocol.py](../lammps_live/remote/protocol.py)) — half a frame of positions
silently reinterpreted would draw a plausible, wrong picture, which is the failure
mode this whole file is organised against.

### Directors: 2 × uint8, octahedral (2 bytes)

A unit vector has three components but only two degrees of freedom, so storing
three wastes a third of the bytes. The octahedral map spends the other two well
([protocol.py:296](../lammps_live/remote/protocol.py#L296)):

```python
def oct_encode(directors, dtype=np.uint16):
    n = np.asarray(directors, dtype=np.float64)
    denom = np.abs(n).sum(axis=1, keepdims=True)      # 1. onto |x|+|y|+|z| = 1
    denom = np.where(denom < 1e-12, 1.0, denom)       #    (a never-set director)
    p = n / denom
    lower = p[:, 2] < 0.0                             # 2. fold the bottom half
    px, py = p[:, 0].copy(), p[:, 1].copy()           #    outward across the
    if lower.any():                                   #    diagonals
        sx = np.where(px[lower] >= 0.0, 1.0, -1.0)
        sy = np.where(py[lower] >= 0.0, 1.0, -1.0)
        px[lower] = (1.0 - np.abs(py[lower])) * sx
        py[lower] = (1.0 - np.abs(px[lower])) * sy
    out = np.empty((len(n), 2), dtype=dtype)          # 3. the square is 2 numbers
    out[:, 0] = quantise(px, -1.0, 1.0, dtype)
    out[:, 1] = quantise(py, -1.0, 1.0, dtype)
    return out
```

Step 1 projects the sphere onto an octahedron. The top half of an octahedron is
already a flat square in (x, y); step 2 reflects the bottom half outward into the
surrounding square, so the whole sphere becomes one unit square. It is continuous
and nearly area-preserving, so no direction is much worse encoded than any other.
`oct_decode` runs the same two steps backwards and renormalises.

Only the *encoder* is told the width. The codes carry it themselves in the
manifest's dtype string, and `oct_decode` reads the scale off `codes.dtype` — so
one decoder handles both codecs with no branch.

Measured over 500,000 random unit vectors, in float64:

| bits/component | max | mean | B/bead |
|---|---|---|---|
| 16 | 0.018° | 0.0045° | 4 |
| **8** | **0.94°** | **0.34°** | **2** |

**This used to be 16, and the argument for it was wrong.** It ran: the client does
not only *draw* these directors, it **measures** `nematic_S` from them, and that
is the number the k_tilt transition shows up in — so an order parameter should not
carry a codec's error. Correct in form, three orders of magnitude out in
magnitude. Measured over 200,000 directors at S ≈ 0.5, coding them at 8 bits moves
S by **8 × 10⁻⁵** — the fourth decimal of a number that is read off a plot.

The reason is worth keeping, because it generalises to every other quantity on
this wire: the angular error is **random and zero-mean**, and S is a **second
moment over the whole population**. Per-bead noise averages *out* of an aggregate
rather than accumulating into it — for N beads it enters as ~θ²  with a 1/√N on
the fluctuation, not as θ. A quantity that summed the directors rather than
averaging their outer product would deserve the caution; this one does not.
(`tests/test_remote_protocol.py` asserts both numbers, so the claim stays honest.)

### Energies: 1 × uint8 (1 byte, on request only)

Per-bead potential energy is quantised over the render style's own colour range,
because that range is exactly the information the client can display — 256 levels
across a colour ramp is more than the eye resolves.

### `raw32` exists for the tests

`CODECS = ("q12", "q16", "raw32")`. `raw32` sends plain float32 (24 B/bead) so the
loopback tests can assert *exact* equality between what the server integrated and
what the client drew. Being able to write `assert_array_equal` is worth more there
than the bandwidth.

---

## 5. The handshake

```
client                                        server
  |                                             |
  |-- connect() ------------------------------->| accept()
  |                                             | settimeout(20s)
  |-- {"t":"hello","version":1,"token":"..."} ->|
  |                                             | version check, compare_digest
  |<- {"t":"building","msg":"..."} -------------| (only on the first client)
  |   settimeout(600s) — this is a wait,        | ~30 s: LAMMPS setup, random fill
  |   not a hang                                |
  |<- {"t":"welcome","natoms":10000,"box":...} -| settimeout(None)
  |                                             |
  |-- {"t":"config",...} {"t":"temp",...} ----->| align the far end with the panel
  |-- {"t":"set","key":"k_tilt",...} ---------->|
  |-- {"t":"pause"} --------------------------->|
  |                                             |
  |<- frame, frame, frame, ... -----------------|
```

Client side ([client.py:93](../lammps_live/remote/client.py#L93)):

```python
sock = socket.create_connection((host, int(port)), timeout=timeout)
protocol.set_socket_options(sock)
sock.sendall(protocol.pack({"t": "hello", "version": protocol.VERSION,
                            "token": token}))
while True:
    header, _payload = protocol.recv_message(sock)
    if header is None or header.get("t") != "building":
        break
    on_notice(str(header.get("msg", "the server is building")))
    sock.settimeout(cls.BUILD_TIMEOUT)          # 600 s — a wait, not a hang
...
sock.settimeout(None)                           # back to blocking forever
```

Server side ([server.py:227](../lammps_live/remote/server.py#L227)):

```python
sock.settimeout(HANDSHAKE_TIMEOUT)              # 20 s
header, _ = protocol.recv_message(sock)
if header.get("version") != protocol.VERSION:
    sock.sendall(protocol.pack({"t": "error", "msg": "protocol version mismatch..."}))
    return None
if not hmac.compare_digest(str(header.get("token", "")), self.token):
    sock.sendall(protocol.pack({"t": "error", "msg": "bad token"}))
    return None
```

Four details that are each there because of a specific failure:

**`settimeout` is toggled three times, and each state matters.** A socket with no
timeout blocks forever. During the handshake that is a denial of service by
accident — the server serves *one* client at a time, so a connection that opens
and says nothing would wedge it permanently, and on a shared cluster network that
is all it takes. So: 20 s for the hello. But once the client is authenticated the
timeout must come **off**, because the same socket is read by the control thread,
which is idle for minutes between slider movements, and a timeout there would read
as a dead client.

**`{"t":"building"}` is sent *before* the build starts, not after.** A build is
`plugin load`, the placement, LAMMPS' own setup, and whatever relaxation the
scenario asks for — all of it before a welcome could be sent, and all of it with
the client sitting in a blocking read. It used to be tens of seconds (almost
entirely LAMMPS' overlap-rejecting random fill, which is now a bulk numpy pass —
see `state.random_points_min_separation`) and is now about a second on the
assembly decks and several on a bonded one that settles. Still long enough to
matter: the client used to give up at 15 s, drop the socket and retry — which made
the server throw the half-built simulation away and start over, so the retry could
not succeed either. That one message is what turns a hang into a wait.

**`hmac.compare_digest`, not `==`.** String comparison short-circuits at the first
differing byte, so its *timing* leaks how many leading bytes you got right — a
token can be guessed one byte at a time. `compare_digest` takes the same time
whatever the input. On a shared cluster network this is not hypothetical.

**The socket options** ([protocol.py:118](../lammps_live/remote/protocol.py#L118)):

```python
def set_socket_options(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
```

`TCP_NODELAY` switches off *Nagle's algorithm*, which by default holds a small
write for up to a round trip hoping to combine it with the next one. Excellent for
a terminal session, terrible here: a control message is one small write, and on a
slider drag the delay is felt directly. `SO_KEEPALIVE` makes the OS probe an idle
connection — the far end is a batch job that can vanish (time limit, node failure,
`scancel`) without ever closing its socket, and without keepalives a blocked read
hangs until the OS gives up minutes later.

---

## 6. Two directions, and why they need three threads

The asymmetry is the whole design: **server → client is megabytes per second,
client → server is a few bytes per second.** All the care goes into the frame
encoding and none into the control path.

```
        SERVER (compute node)                    CLIENT (laptop)
  ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │ main thread: serve_client    │        │ stepper worker: step()        │
  │   drain control queue        │        │   take_frame() → _ingest()    │
  │   system.step(20)            │──────▶ │   → analysis (~10 ms)         │
  │   encode_frame()             │ frames │                               │
  │   sendall()                  │        │ frame-reader thread:          │
  │                              │        │   recv_message() forever      │
  │ control-reader thread:       │ ◀──────│   keeps only the NEWEST frame │
  │   recv_message() → Queue     │control │                               │
  └──────────────────────────────┘        │ main thread: pygame, 60 fps   │
                                          └──────────────────────────────┘
```

### Server: the control reader is its own thread

```python
class ControlChannel:
    def __init__(self, sock):
        self.messages = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                header, _payload = protocol.recv_message(self.sock)
                if header is None:
                    break
                self.messages.put(header)
        except (OSError, protocol.ProtocolError):
            pass
        finally:
            self.closed.set()

    def drain(self):                    # non-blocking: everything queued so far
        out = []
        while True:
            try:
                out.append(self.messages.get_nowait())
            except queue.Empty:
                return out
```

Reading and sending have nothing to say to each other. A frame `sendall` may block
for as long as the client takes to drain its buffer — which is the backpressure
that keeps the whole thing honest — and a slider change must not wait behind it.
`queue.Queue` is thread-safe, so no lock is written by hand.

The serve loop then reads like an ordinary game loop
([server.py:259](../lammps_live/remote/server.py#L259)):

```python
while not control.closed.is_set() and not self._stop.is_set():
    if self._apply_control(control.drain(), sock):
        return                                  # the client said "bye"
    if self.playing:
        system.step(self.steps_per_frame)       # 20 MD steps on the A100
    self._flush_fault(sock)
    self._send_frame(sock, system)
    if self.fps > 0:
        next_send = max(next_send + 1.0 / self.fps, time.monotonic() - 0.25)
        delay = next_send - time.monotonic()
        if delay > 0:
            control.closed.wait(delay)          # sleep, but wake on a disconnect
```

That last line is a small idiom worth stealing: `Event.wait(delay)` is a sleep that
returns early if the event fires, so a paused server does not sit out a quarter of
a second past a disconnect.

### Client: the reader thread keeps a queue of one

```python
def _read_loop(self):
    while True:
        header, payload = protocol.recv_message(self.sock)
        kind = header.get("t")
        if kind == "frame":
            with self._lock:
                if self._latest is not None:
                    self.dropped += 1           # overwrite: the old one is stale
                self._latest = (header, payload)
                self.received += 1
            self._new.set()
        elif kind == "fault":
            self.faults.append(header)          # queued — must NOT be dropped
        elif kind == "pong":
            ...

def take_frame(self, timeout=0.25):
    """The newest frame, or None if none arrived inside `timeout`."""
    self._new.wait(timeout)
    with self._lock:
        frame, self._latest = self._latest, None
        if frame is None:
            self._new.clear()
    return frame
```

**Frames are dropped, never queued.** If this machine falls behind — a hitch, a
slow analysis frame, a window resize — the next `step()` picks up where the
simulation actually *is*, not where it was three frames ago. A queue would trade
latency for smoothness, and over a link the renderer cannot flow-control, the
latency would only ever grow. The drop count is on the HUD.

**Faults are the exception and go in a real (bounded) deque.** They are the one
message that must not be dropped, and the latest-frame slot exists precisely to
drop things. This was originally a field on the frame header, which was wrong for
a reason worth remembering: on a link fast enough to drop frames — the whole point
of the A100 — the one frame carrying the event is usually the one thrown away. An
event that is only sometimes delivered is worse than none.

**`step(n)` ignores `n`.** How far the simulation advances per frame is the
server's decision, and it reports it back as `dt` on each frame
([client.py:491](../lammps_live/remote/client.py#L491)):

```python
def step(self, n):
    if not self.connected:
        time.sleep(1.0 / 60)                    # don't burn a core
        return
    self._sync_energy_request()
    self._ping_countdown -= 1
    if self._ping_countdown <= 0:
        self._ping_countdown = 60
        self.link.ping()
    t0 = time.perf_counter()
    frame = self.link.take_frame(timeout=0.25)
    self.wait_seconds = time.perf_counter() - t0
    if frame is not None:
        self._ingest(frame)
```

This runs on the *stepper's worker thread*, not the render thread — so the network
wait **and** the analysis that follows it overlap the drawing of the previous
frame. That is not incidental: the analysis is the expensive half of this end
(~1.5 µs/bead/chunk, so ~10 ms at 10k beads), and overlapped it costs
`max(analysis, render)` per frame rather than the sum.

### The control path back

Sending is one method, and it deliberately never raises
([client.py:195](../lammps_live/remote/client.py#L195)):

```python
def send(self, message):
    if self.closed.is_set():
        return False
    try:
        with self._send_lock:
            self.sock.sendall(protocol.pack(message))
        return True
    except OSError as exc:
        self.error = self.error or f"send failed: {exc}"
        self.closed.set()
        return False
```

The `_send_lock` matters: `sendall` from two threads at once would interleave two
messages' bytes and corrupt the stream. The reader thread is the single authority
on whether the link is up, so a slider hitting a dead socket returns `False`
instead of throwing an exception into the UI event loop.

The message vocabulary is small — `_apply_one`
([server.py:377](../lammps_live/remote/server.py#L377)):

| message | effect on the far side |
|---|---|
| `{"t":"set","key":"k_tilt","value":4.0}` | `system.set_extra_param(...)` — the same call the local app makes |
| `{"t":"temp","value":0.2}` | `system.set_target_temp(...)` |
| `{"t":"play"}` / `{"t":"pause"}` | flips `self.playing` |
| `{"t":"reset"}` | `system.reset()`, and `seq` restarts at 0 |
| `{"t":"config","fps":30,"codec":"q12","energies":true}` | frame configuration |
| `{"t":"ping","id":42}` | answered with `{"t":"pong","id":42}` |
| `{"t":"bye"}` | end the session |

Every one of these is *the same call the local app makes on its own system*. The
control channel is a remote procedure call onto the `MDSystem` interface, not a
second way of doing things.

`ping`/`pong` is answered **on the send thread**, deliberately, so the round trip
the client measures includes any time spent waiting behind a frame — which is the
number that actually matters for how a slider feels. The client fires one every 60
frames and shows the result on the HUD:

```python
def ping(self):
    ping_id = self.received + 1
    self._pings[ping_id] = time.monotonic()
    if len(self._pings) > 8:            # a link that stopped answering
        self._pings.clear()
    self.send({"t": "ping", "id": ping_id})
```

**A slider must not cost the allocation.** Anything that reaches LAMMPS can be
refused by it, and letting that exception propagate would end the server process —
which runs `scancel` on its own job on the way out. So `_apply_control` catches
everything, turns it into a `Fault` message, and keeps going.

---

## 7. What `_ingest` does with a frame

```python
def _ingest(self, frame):
    header, payload = frame
    codec = header.get("codec", protocol.DEFAULT_CODEC)
    arrays = protocol.decode_frame(
        header.get("arrays") or [], payload, self.box,
        energy_range=self.spec.render_style.energy_range, codec=codec)
    positions = arrays.get("positions")
    if positions is None:
        return
    self._state = FrameState(positions=positions,
                             directors=arrays.get("directors"),
                             types=None, ids=self.all_ids, box=self.box)
    self._energies = arrays.get("energies")
    seq = int(header.get("seq") or 0)
    if self._resetting and seq <= self._seq:    # the first frame of a NEW run
        self._resetting = False
    self._seq = seq
    self._sim_time = float(header.get("sim_time") or 0.0)
    self._last_step_dt = float(header.get("dt") or 0.0)
    ...
    self.analysis.update(self._state, self.params)      # ~10 ms at 10k beads
```

Note what is *not* here: no interpolation against a previous frame, no state
carried between frames, nothing that a missing frame would invalidate. Every frame
decodes entirely on its own. That is the property that lets the reader thread
throw frames away freely.

`self.box` comes from the welcome message and is authoritative — the client also
computes a box locally from the scenario (so the camera can frame the scene before
a single frame arrives), but the server's replaces it on connect.

The `seq <= self._seq` test is the only place the sequence number is used: after a
Reset, the server restarts its numbering at 0, so a frame numbered at or below the
one we were on is the first of the new run. Without it, the picture sits on the
last frame of the *old* run, which is indistinguishable from a Reset that did
nothing.

---

## 8. The other half of the bandwidth: sending fewer frames than you draw

Everything above makes each frame smaller. This makes there be fewer of them, and
it is the bigger lever of the two: at 50k beads, `q12` at 60 fps is 19.5 MB/s and
at 20 fps it is 6.5. Over an SSH tunnel from Amsterdam that is the difference
between a link that keeps up and one that does not.

Three things have to be true before a slow wire is usable, and each one is a place
this was originally wrong.

### 8a. The simulation must not slow down with the wire

A scenario declares `sim_time_per_frame` — how much simulated time one **drawn**
frame should advance — and it is a number somebody chose by watching. Send at 20
fps with the same stride and the demo runs at a third speed.

The obvious fix is `--free-run`: integrate flat out between sends, which also uses
the GPU that is otherwise idle. **It is a trap, and worth understanding why.** On
an A100 at 50k beads a stride takes well under a millisecond, so free-running for
a 50 ms send interval advances *1,420 steps instead of 60*:

| | steps/frame | pace | GPU busy |
|---|---|---|---|
| 60 fps, one stride | 20 | 12 τ/s | 4% |
| 20 fps, `--free-run` | ~1,420 | **284 τ/s** | 99% |
| 20 fps, stride ×3 | 60 | 12 τ/s | 4% |

284 τ/s is not the demo — assembly is over before anyone has looked at it, and
every slider acts on a system that has already moved on. Worse for this document's
purposes, consecutive frames then land ~14 τ apart, which is far enough that they
are **nearly uncorrelated**. Everything in §8c depends on two consecutive frames
being two views of the same arrangement, and free-run destroys that premise along
with the trajectory smoothing's.

So the stride is derived from the send rate instead
([server.py](../lammps_live/remote/server.py), `_stride_for`):

```python
def _stride_for(self, fps):
    if self.steps_override:
        return max(1, int(self.steps_override))
    scenario = self.system.scenario
    per_frame = max(1, round(scenario.sim_time_per_frame / scenario.timestep))
    if fps <= 0:
        return per_frame
    return max(1, round(per_frame * self.REFERENCE_FPS / float(fps)))
```

Measured against a real server, varying only the client's requested rate:

| target fps | stride | frames/s | sim pace |
|---|---|---|---|
| 60 | 20 | 60.0 | 12.0 τ/s |
| 30 | 40 | 30.0 | 12.0 τ/s |
| 20 | 60 | 20.0 | 12.0 τ/s |
| 10 | 120 | 10.0 | 12.0 τ/s |

**And the idle GPU is fine.** The A100 is here to make 50k beads possible at all,
not to maximise steps per second; the allocation costs the same whether it is 4%
busy or 99%.

### 8b. The window must not be slaved to the wire

`App._tick` opens by waiting for the step launched under the previous frame's
drawing, and for a remote system that "step" is `RemoteSystem.step()`, which waited
for a frame. So *the app loop ran at the wire's rate* — and no amount of filling in
between frames can help if there are no frames in between to fill.

The fix is one constant. `step()` polls briefly and returns with nothing, which is
the normal case rather than a failure:

```python
FRAME_POLL = 0.006

frame = self.link.take_frame(timeout=self.FRAME_POLL)
```

Not zero, because a stepper that returns instantly spins a core; short enough that
a frame arriving during one poll is picked up on the very next tick. At 20 fps
sent and 60 drawn, two ticks in three now return with nothing and draw whatever
`_render_state` gives them.

### 8c. Something has to move between frames

Otherwise a 60 Hz window draws a 20 Hz slideshow, and thermal motion is exactly
the signal the eye reads as "frozen" the moment it steps.

The tempting answer is to predict the missing frames. Measured on a real
1500-bead trajectory, replaying a 20 Hz wire against the 60 Hz truth (mean
per-bead error, in screen pixels):

| scheme | smoothing off | smoothing τ=1.2 |
|---|---|---|
| hold the newest frame (a slideshow) | 10.4 px | 5.0 px |
| interpolate between the last two | **5.0 px** | **1.0 px** |
| extrapolate from the last two | 11.2 px | 2.8 px |

**Extrapolation is worse than freezing the picture.** That result is the whole
design: between wire frames the motion is dominated by thermal rattle, which is
uncorrelated frame to frame, so a velocity estimated from two samples is a random
number and integrating it doubles the error rather than reducing it. Only
interpolation beats holding — and it beats it by *bracketing* the truth, which
means holding the newest frame back and playing one wire frame behind. That is
50 ms of latency at 20 Hz, paid on every slider, forever.

So the shipped answer buys fluency instead of accuracy, at zero latency:
[`playground/jitter.py`](../lammps_live/playground/jitter.py) synthesises the
rattle rather than predicting it. An overdamped Langevin particle's excursion
about a slowly-moving centre *is* an Ornstein-Uhlenbeck process, so that is what
it generates — mean-reverting, temporally correlated, not the white noise that
would read as television static.

It should be read as a cosmetic device. It does not reduce the error against the
true trajectory; it slightly increases it. What it buys is that the picture never
looks frozen.

**The amplitude is measured, not declared.** The median per-bead displacement
between two received frames is real motion in the right units, and deriving from
it is what makes the effect go quiet on its own exactly where it should: at
temperature zero, while paused, and on a system that has stopped rearranging.

Converting that measurement into the OU's `sigma` is the one piece of arithmetic
worth reading, because the naive version is wrong three times over and *every
correction lands near 1 at the 20-sent/60-drawn default* — so a hardcoded constant
would have looked right on one machine and been silently wrong on every other:

1. **The measurement is a length; the process wants a component.** The median of
   |v| for an isotropic Gaussian is `sigma * sqrt(median of chi-squared_dof)` —
   1.538 for a position's three degrees of freedom, 1.177 for the two a unit
   director has left after renormalisation. Skipping this alone runs the scene
   54% hot, which is exactly how the first draft shipped.
2. **A drawn frame is a fraction of a wire frame.** A full-rate wire would have
   shown `wire_step * sqrt(share)`. The square root is not a fudge: these are
   overdamped beads, so displacement grows as √t.
3. **An OU step is not `sigma`.** Consecutive samples differ by
   `sigma * sqrt(2*(1 - rho))`, and `rho` depends on the frame time — so
   inverting it is what keeps the apparent temperature the same at 30 fps as at
   60, instead of making a slow machine look cold.

**What it does not hide.** The motion a slow wire loses is not *missing*, it is
**lumpy**: a 20 Hz frame carries three frames' worth of displacement and lands all
of it on one drawn frame in three. Filling the other two to full motion therefore
leaves that one moving more than the rest, and the picture carries ~1.4× the true
motion overall. That excess *is* the residual stutter — now under a surface that
never freezes rather than on top of one that does. Removing it means easing toward
each arriving frame instead of snapping to it, which is latency again.

Both properties are asserted in `tests/test_jitter.py`, driven from a synthetic
diffusing population where the true per-frame step is known exactly, so neither
can quietly change.

Two smaller consequences of drawing more often than receiving:

- **`_render_state` now runs at the screen's rate, not the wire's**, so it can no
  longer be a straight cache on the received frame. It is keyed on wall time as
  well, because it is called two or three times per drawn frame (positions,
  directors, the 2D readout) and must advance the rattle exactly once.
- **The smoothing filter had to be told.** Its weight comes from how much
  *simulated* time a frame advanced; applied once per drawn frame instead of once
  per received one, it would take three full steps per frame of physics and smooth
  three times as hard as the slider says. So it is handed this frame's share.

The two filters compose in one order, and it is the order that matters: **rattle
first, smooth second**, so that the Smoothing slider removes the synthetic motion
exactly as it removes the real kind.

---

## 9. The tunnel: how a socket to `127.0.0.1` reaches a GPU node

Everything above assumes the client can `connect()` to the server. It cannot —
`gcn12` is on Snellius' internal network, there is no route from your laptop and
there is a firewall, and both of those are correct.

SSH carries the socket. What `-L` does, precisely:

```
ssh -L 5723:127.0.0.1:5723  user@gcn12
```

tells the *local* ssh process: **listen on 127.0.0.1:5723 on this machine; for
every connection that arrives there, open a channel over the SSH session and ask
the far end to connect to `127.0.0.1:5723` as seen from where the session ends.**
Then relay bytes both ways. Your `socket.create_connection(("127.0.0.1", 5723))`
is talking to the local ssh process, which is impersonating the server.

**Where the session ends is the whole question**, and it is why this codebase
defaults to two hops ([target.py:52](../lammps_live/remote/target.py#L52)):

| | `tunnel="forward"` (one hop) | `tunnel="jump"` (default) |
|---|---|---|
| session terminates on | the login node | **the compute node** |
| frames on the internal network | **plain TCP** — sshd decrypts and re-opens | **encrypted end to end** |
| server binds | `0.0.0.0` | **`127.0.0.1`** |
| who can reach the port | anything on the cluster network | **only processes on that node** |

The one-hop form is the obvious one and it is a real exposure: the login node's
sshd decrypts your stream and opens a *separate plain TCP connection* onward, so
the frames cross the internal network in clear text and the server has to listen
on an interface every other user can reach — with a control channel that can
re-issue LAMMPS commands behind nothing but a shared secret.

The two-hop command, as built at
[session.py:972](../lammps_live/remote/session.py#L972):

```python
proxy = (f"ssh -S {shlex.quote(self._control_path)} "
         f"-o ControlMaster=no -W %h:%p {self.target.destination}")
cmd = ["ssh", "-N", "-T",
       "-o", f"ProxyCommand={proxy}",
       "-o", "ControlMaster=no",
       "-o", "ControlPath=none",
       "-o", "ExitOnForwardFailure=yes",
       "-o", f"UserKnownHostsFile={known_hosts}",
       "-o", "StrictHostKeyChecking=accept-new",
       "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=4",
       "-L", f"{candidate}:127.0.0.1:{self._remote_port}", destination]
```

- **`-W %h:%p`** makes the jump hop a plain stdio pipe over the *already
  authenticated* master connection, so the login node relays an opaque stream and
  no second login is needed. (`%h`/`%p` are substituted by ssh itself before the
  shell sees the string, which is why they can sit inside an f-string unquoted.)
- **`ControlPath=none`** cost a whole debugging session. A `~/.ssh/config` with
  `ControlMaster auto` + `ControlPersist` for the cluster — a sensible thing for a
  human to have — changes what `ssh -N -L` *means*: it backgrounds the master and
  exits 0. The port is bound for a moment, so the forward looks like it came up,
  and then the process holding it is gone. The failure reaches the user as "the
  tunnel is open but the server did not answer", which is true and useless.
- **`ExitOnForwardFailure=yes`** so a port that cannot be bound is a loud failure,
  not a live connection carrying nothing.
- **Host keys go in a session-local file.** Compute nodes are reimaged and their
  keys change; pinning them in the real `known_hosts` turns a routine node
  reallocation into a scary warning weeks later.

And because `ssh -N` reports success by *continuing to run* — while already
running before the forward exists — the only honest readiness signal is polling
the port, with the process checked at the same time
([session.py:1083](../lammps_live/remote/session.py#L1083)):

```python
def _await_forward(self, proc, port, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exited = proc.poll() is not None
        if self._port_in_use(port) and not exited:
            return True
        if exited:
            return False                # backgrounded itself, or failed
        time.sleep(0.2)
    return False

@staticmethod
def _port_in_use(port):
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0
```

`connect_ex` returns an errno instead of raising — the idiomatic way to *test* a
port rather than use it.

### The token never appears in a command line

The server refuses to run without a shared secret, because its control channel can
re-issue LAMMPS commands. But on a shared cluster every user can read every other
user's `ps` output, so `--token <secret>` would publish it to the whole login node.
Instead it goes down the SSH channel's stdin, which nothing else can see
([session.py:876](../lammps_live/remote/session.py#L876)):

```python
self._server_proc = subprocess.Popen(argv, stdin=subprocess.PIPE, ...)
self._server_proc.stdin.write(self._token + "\n")
self._server_proc.stdin.flush()
```

and on the far side, `--token-stdin` reads exactly one line
([server.py](../lammps_live/remote/server.py)). `tests/test_remote_session.py`
asserts the token never appears in any command line the session builds.

### How the client learns the port

The server prints one line the moment it is listening, and the session watches
for it:

```python
self.log(f"LISTENING host={socket.gethostname()} port={self.port}")
```

```python
match = re.search(r"LISTENING host=(\S+) port=(\d+)", line)
```

That is the synchronisation point for the whole flow: `srun`'s stdout is an SSH
pipe the GUI is reading, and that line is what says the tunnel has something to
reach. (Which is also why every `log` call passes `flush=True` — an unflushed line
is a line the user never sees.)

---

## 10. Run the whole thing yourself

### The 50-line version — no LAMMPS, no SSH, one socket

Save as `toy.py`, run with `./venv/bin/python toy.py`:

```python
"""A stand-in for the whole pipeline: no LAMMPS, no SSH, one socket."""
import socket, threading, time
import numpy as np
from lammps_live.remote import protocol
from lammps_live.playground.state import Box, FrameState

BOX = Box((0.0, 0.0, 0.0), (20.0, 20.0, 20.0), (True, True, True))
N, TOKEN = 500, "dev"


def server(sock):
    header, _ = protocol.recv_message(sock)          # 1. the hello
    assert header["t"] == "hello" and header["token"] == TOKEN
    sock.sendall(protocol.pack({"t": "welcome", "natoms": N,
                                "box": {"lo": list(BOX.lo), "hi": list(BOX.hi)}}))
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 20, size=(N, 3))
    for seq in range(3):                             # 2. three "frames"
        pos = (pos + 0.05) % 20.0                    #    "integrate"
        state = FrameState(positions=pos, directors=None, types=None,
                           ids=np.arange(1, N + 1), box=BOX)
        manifest, payload = protocol.encode_frame(state, BOX, codec="q12")
        sock.sendall(protocol.pack({"t": "frame", "seq": seq, "n": N,
                                    "codec": "q12", "arrays": manifest}, payload))
        time.sleep(0.01)


def client(host, port):
    sock = socket.create_connection((host, port))
    protocol.set_socket_options(sock)
    sock.sendall(protocol.pack({"t": "hello", "version": protocol.VERSION,
                                "token": TOKEN}))
    welcome, _ = protocol.recv_message(sock)
    print("welcome:", welcome)
    for _ in range(3):
        header, payload = protocol.recv_message(sock)
        arrays = protocol.decode_frame(header["arrays"], payload, BOX, codec="q12")
        print(f"frame {header['seq']}: {len(payload)} B payload, "
              f"bead 0 at {arrays['positions'][0]}")
    sock.close()


listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))          # port 0 = let the OS pick a free one
listener.listen(1)
port = listener.getsockname()[1]
threading.Thread(target=lambda: server(listener.accept()[0]), daemon=True).start()
client("127.0.0.1", port)
```

```
welcome: {'t': 'welcome', 'natoms': 500, 'box': {...}}
frame 0: 2250 B payload, bead 0 at [12.790965   5.4468865  0.8695971]
frame 1: 2250 B payload, bead 0 at [12.839316   5.4952383  0.9179487]
frame 2: 2250 B payload, bead 0 at [12.887669   5.54359    0.9716728]
```

`SO_REUSEADDR` is there because a TCP port sits in `TIME_WAIT` for a minute or so
after a close; without it, restarting a server immediately fails with "address
already in use".

### The real pipeline, on this machine

Everything except SSH and Slurm — a real `FrameServer`, a real socket, the real
`RemoteSystem`, just both ends on your laptop:

```bash
./venv/bin/python -m lammps_live.remote.server --playground mesomem_remote \
       --profile local --token dev --port 5723
./venv/bin/lammps-live --playground mesomem_remote --remote 127.0.0.1:5723 --token dev
```

### The tests

- `tests/test_remote_protocol.py` — the codec alone: round-trip error bounds,
  truncated payloads, awkward directors.
- `tests/test_remote_loopback.py` — the whole pipeline over a real socket with a
  real 900-bead LAMMPS: a slider reaching the far end, frames being dropped and
  not queued, Reset rebuilding *there*, every readout safe before the first frame
  and after the link drops.
- `tests/test_remote_session.py` + `tests/fake_cluster/` — the SSH and Slurm half
  against a fake `ssh`/`salloc`/`squeue`/`scancel` that stands in for the machine
  boundary. Not mocks: the tar is really unpacked, the server really starts, the
  token really arrives on stdin, and the forwarded port really carries frames.

---

## 11. Things that will bite you, in one list

| symptom | cause |
|---|---|
| garbage after a while | you assumed `recv(n)` returns `n` bytes. It does not. |
| two messages merged / one split | TCP has no messages. Length-prefix everything. |
| a slider feels laggy but frames are fine | Nagle. `TCP_NODELAY`. |
| a dead peer takes minutes to notice | no `SO_KEEPALIVE`, or no `ServerAliveInterval` on the ssh hop. |
| "address already in use" on restart | `TIME_WAIT`; set `SO_REUSEADDR`. |
| corrupted stream when two threads send | `sendall` is not atomic across threads. One send lock. |
| the server hangs forever on one bad client | a blocking `recv` with no handshake timeout. |
| latency that only ever grows | you queued frames instead of dropping them. |
| "the tunnel is open but the server did not answer" | the ssh backgrounded itself (`ControlPersist`), or the forward went to a node the task is not on. |
| a token guessable byte by byte | `==` instead of `hmac.compare_digest`. |

---

## See also

- [remote-gpu.md](remote-gpu.md) — the design: why two hops, why `salloc
  --no-shell`, why `SSH_ASKPASS`, the eight real bugs, and what to do next.
- [a100-plan.md](a100-plan.md) §5 — the measured compression table, including the
  schemes deliberately *not* built (temporal deltas, zlib) and why.
- [snellius/README.md](snellius/README.md) — how to actually run it.
