# Cross-compilation setup and gotchas

Target: Embedsky E9V3 (i.MX6Q, ARMv7) running Linux 4.1.
Host: Ubuntu 14.04, Linaro GCC 5.3 (`arm-linux-gnueabihf`).

This is the part that took the longest and is worth reading if you are doing anything
similar. Everything below was found the hard way.

---

## 1. Toolchain

Install the Linaro cross compiler to `/opt/EmbedSky/`, then verify:

```bash
arm-linux-gnueabihf-gcc -v
```

The toolchain root used throughout is:

```
/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf
```

**Gotcha — GCC 5.3 defaults to C++98.** Anything written against C++11 will fail to
compile. Concretely, these had to be rewritten:

```cpp
// C++11 — will not compile
std::map<int, std::string> m = {{312553011, "312553011"}};
for (const auto& r : faces) { ... }

// C++98 — what actually ships
std::map<int, std::string> m;
m[312553011] = "312553011";
for (std::vector<cv::Rect>::const_iterator it = faces.begin(); it != faces.end(); ++it) {
    const cv::Rect& r = *it;
    ...
}
```

You can pass `-std=c++11`, but the bundled libstdc++ on the target is old enough that it is
less trouble to just write C++98.

---

## 2. Cross-compiling OpenCV 3.4.7

Configure with `cmake-gui`, selecting **"Specify options for cross-compiling"**
(OS: `Linux`, version `4.1`, processor `arm`), pointing the C/C++ compilers at
`arm-linux-gnueabihf-gcc` / `-g++`.

Key settings:

| Option | Value |
|---|---|
| `CMAKE_INSTALL_PREFIX` | `/usr/local/arm-opencv/install` |
| `BUILD_opencv_world` | ON (single `.so`, far simpler to deploy) |
| `OPENCV_FORCE_3RDPARTY_BUILD` | ON |
| `ENABLE_CXX11` | ON |
| `BUILD_PERF_TESTS`, `BUILD_TESTS`, `BUILD_opencv_ts` | OFF |
| `BUILD_opencv_python_bindings_generator`, `..._python_tests` | OFF |
| All `WITH_*` | OFF, **except** `WITH_V4L` |

### Gotcha — Qt breaks the DNN module

The course instructions have you enable `WITH_QT` (and set `Qt5_DIR`). That is fine for
Lab 2, but **once you need `cv::dnn` for YOLO, the build fails.** Qt support and the DNN
module conflict during linking.

**Fix: turn `WITH_QT` off and rebuild.** You lose `cv::imshow`, which does not matter here
because everything renders through the framebuffer anyway.

If you only ever need Lab 2, keep Qt and set:

```
Qt5_DIR = /opt/EmbedSky/.../qt5.5/rootfs_imx6q_V3_qt5.5_env/qt5.5_env/lib/cmake/Qt5
```

### Build

```bash
cd /usr/local/arm-opencv/build
sudo make -j$(nproc)
sudo make install
```

---

## 3. Compiling against it

```bash
arm-linux-gnueabihf-g++ source.cpp -o demo \
  -I /opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/include/ \
  -I /usr/local/arm-opencv/install/include/ \
  -L /usr/local/arm-opencv/install/lib/ \
  -Wl,-rpath-link=/opt/EmbedSky/.../arm-linux-gnueabihf/libc/lib/ \
  -Wl,-rpath-link=/opt/EmbedSky/.../qt5.5/rootfs_imx6q_V3_qt5.5_env/lib/ \
  -Wl,-rpath-link=/opt/EmbedSky/.../qt5.5/rootfs_imx6q_V3_qt5.5_env/usr/lib/ \
  -lpthread -lopencv_world
```

The `-rpath-link` flags exist because the linker must resolve the *transitive* dependencies
of `libopencv_world.so` (libc, libstdc++, Qt libs) at link time, and those live in the
target sysroot rather than on the host.

---

## 4. Deploying to the board

```bash
cp /usr/local/arm-opencv/install/lib/libopencv_world.so.3.4.7 <SD card>
# the runtime linker looks for the SONAME, which is truncated:
mv libopencv_world.so.3.4.7 libopencv_world.so.3.4
```

Run with the library path pointing at the current directory:

```bash
LD_LIBRARY_PATH=. ./demo
```

Plain `./demo` fails with "cannot open shared object file" — the board has no OpenCV in
`/usr/lib` and no `ldconfig` entry for it.

---

## 5. Framebuffer gotchas

### 5.1 Stride is not `width × bpp`

The single most costly bug in this project. Output looked skewed or garbled on some
display configurations because the code assumed each row was `xres_virtual × bpp/8` bytes.

Framebuffer drivers pad rows for alignment. The real value is `line_length` from
`FBIOGET_FSCREENINFO`:

```cpp
struct fb_var_screeninfo screen_info;   // resolution, bpp
struct fb_fix_screeninfo fix_info;      // <-- the fixed info, including line_length

ioctl(fbfd, FBIOGET_VSCREENINFO, &screen_info);
ioctl(fbfd, FBIOGET_FSCREENINFO, &fix_info);

uint32_t line_pitch = fix_info.line_length;   // NOT xres_virtual * (bpp/8)
```

Every row offset must then be `y * line_pitch`, and the buffer size
`line_pitch * yres_virtual`.

### 5.2 Rendering, not inference, was the first bottleneck

The naive loop does one `seekp()` + one `write()` per row — 1080 syscall pairs per frame:

```cpp
for (int y = 0; y < height; y++) {
    ofs.seekp(y * line_pitch);
    ofs.write(row_ptr, row_bytes);        // slow: syscall per row
}
```

Building the whole frame in a userspace buffer and flushing it once is dramatically
faster:

```cpp
std::fill(screen_buffer.begin(), screen_buffer.end(), 0);   // letterbox bars for free
char* start = screen_buffer.data() + y_offset * line_pitch + x_offset * bytes_per_pixel;
for (int y = 0; y < scaled_height; y++)
    memcpy(start + y * line_pitch, frame.ptr(y), video_line_bytes);

ofs.seekp(0);
ofs.write(screen_buffer.data(), screen_buffer.size());      // one syscall per frame
```

This also makes the black letterbox/pillar bars free — clearing the buffer handles them,
instead of writing three separate black regions per row.

### 5.3 Colour depth varies by output

The LCD is 16-bit, HDMI is often 32-bit. Branch on `bits_per_pixel` rather than hardcoding
`BGR565`:

```cpp
switch (fb_info.bits_per_pixel) {
    case 16: cv::cvtColor(src, dst, cv::COLOR_BGR2BGR565); break;
    case 24: dst = src; break;                                   // already BGR888
    case 32: cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);  break;
}
```

### 5.4 The framebuffer may be asleep

The Linux framebuffer is in sleep/blank mode by default on this board. HDMI output also
needs u-boot boot arguments changed — see the `TQIMX6_uboot` manual.

Quick sanity check that the device works at all:

```bash
cat /dev/urandom > /dev/fb0    # screen fills with noise
```

---

## 6. Non-blocking keyboard input

`getchar()` blocks and would stall the video loop. Put the terminal in raw mode and poll
with `select()`:

```cpp
int kbhit(void) {
    struct timeval tv = {0L, 0L};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}
```

Remember to save `termios` on entry and restore it on exit, otherwise the shell is left
with echo disabled after the program quits.

---

## References

- Linux framebuffer API — <https://www.kernel.org/doc/Documentation/fb/api.txt>
- Armadeus framebuffer notes — <http://www.armadeus.org/wiki/index.php?title=Framebuffer>
- `ioctl(2)` — <https://man7.org/linux/man-pages/man2/ioctl.2.html>
