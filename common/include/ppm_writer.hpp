#pragma once
#include <string>

struct Pixel;           // AOS
struct FramebufferSOA;  // SOA

// escribe framebuffer SOA a archivo PPM en formato P3
bool writePPM_SOA(std::string const & ruta, FramebufferSOA const & fb, int ancho, int alto);
