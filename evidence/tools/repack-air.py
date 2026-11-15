import hashlib
from pathlib import Path
import struct
import sys

source, bitcode, output = map(Path, sys.argv[1:])
original = bytearray(source.read_bytes())
assert original[:4] == b'MTLB'
offset, old_size = struct.unpack_from('<QQ', original, 72)
raw = bitcode.read_bytes()
if raw[:4] == b'\xde\xc0\x17\x0b':
    raw_offset, raw_size = struct.unpack_from('<II', raw, 8)
    raw = raw[raw_offset:raw_offset + raw_size]
assert raw[:4] == b'BC\xc0\xde'
wrapped = struct.pack('<5I', 0x0B17C0DE, 0, 20, len(raw), 0xFFFFFFFF) + raw
wrapped += bytes(-len(wrapped) % 16)
delta = len(wrapped) - old_size
new = original[:offset] + wrapped + original[offset + old_size:]
struct.pack_into('<Q', new, 16, len(new))
struct.pack_into('<Q', new, 80, len(wrapped))
for tag in [b'MDSZ\x08\x00']:
    at = new.index(tag)
    struct.pack_into('<Q', new, at + len(tag), len(wrapped))
at = new.index(b'HASH\x20\x00') + 6
new[at:at + 32] = hashlib.sha256(wrapped).digest()
# Extended reflection section descriptors follow the function-list ENDT.
at = original.index(b'ENDT', 88) + 4
public_offset = struct.unpack_from('<Q', original, 40)[0]
while at + 6 <= public_offset:
    tag = original[at:at + 4]
    size = struct.unpack_from('<H', original, at + 4)[0]
    if at + 6 + size > public_offset:
        break
    if tag in [b'HSRP', b'HDYN', b'RLST', b'SLST'] and size == 16:
        value = struct.unpack_from('<Q', original, at + 6)[0]
        if value >= offset + old_size:
            struct.pack_into('<Q', new, at + 6, value + delta)
    at += 6 + size
output.write_bytes(new)
print(f'{output}: {len(new)} bytes, AIR {len(raw)} bytes, delta {delta}')
