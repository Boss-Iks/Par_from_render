#include "ppm_writer.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>

struct FramebufferSOA {
  std::vector<std::uint8_t> R;
  std::vector<std::uint8_t> G;
  std::vector<std::uint8_t> B;
};

namespace {

  // deleter personalizado para FILE* usado con unique_ptr
  struct FileCloser {
    void operator()(std::FILE * archivo) const noexcept {
      if (archivo != nullptr) {
        // NOLINTNEXTLINE(cert-err33-c,cppcoreguidelines-owning-memory)
        (void) std::fclose(archivo);
      }
    }
  };

  // alias de tipo para unique_ptr que maneja FILE* con RAII
  using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

  // abre archivo para escritura con manejo RAII
  [[nodiscard]] FilePtr abrir_archivo(std::string const & ruta) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    std::FILE * archivo = std::fopen(ruta.c_str(), "w");
    if (archivo == nullptr) {
      throw std::runtime_error(std::string("fopen fallo: ") + std::strerror(errno));
    }
    return FilePtr(archivo);
  }

  // escribe encabezado PPM con formato P3
  void escribir_encabezado(std::FILE * archivo, int ancho, int alto) {
    std::println(archivo, "P3\n{} {}\n255", ancho, alto);
  }

}  // namespace

// escribe framebuffer SOA a archivo PPM (con paralelismo seguro)
bool writePPM_SOA(std::string const & ruta, FramebufferSOA const & fb, int ancho, int alto) {
  auto const archivo = abrir_archivo(ruta);
  escribir_encabezado(archivo.get(), ancho, alto);

  // buffer de texto por fila: cada hilo escribe en filas distintas (sin carreras)
  std::vector<std::string> filas_texto(static_cast<std::size_t>(alto));

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<int>(0, alto),
      [&](oneapi::tbb::blocked_range<int> const & range) {
        for (int fila = range.begin(); fila != range.end(); ++fila) {
          std::ostringstream oss;

          auto const indice_fila = static_cast<std::size_t>(fila) * static_cast<std::size_t>(ancho);

          for (int col = 0; col < ancho; ++col) {
            auto const idx = indice_fila + static_cast<std::size_t>(col);
            // mismo formato que escribir_triplete: "r g b\n"
            oss << static_cast<unsigned>(fb.R[idx]) << ' ' << static_cast<unsigned>(fb.G[idx])
                << ' ' << static_cast<unsigned>(fb.B[idx]) << '\n';
          }

          filas_texto[static_cast<std::size_t>(fila)] = std::move(oss).str();
        }
      },
      oneapi::tbb::simple_partitioner{});  // puedes cambiar a static_partitioner o auto_partitioner

  // escritura secuencial al FILE* en el orden correcto de filas
  for (int fila = 0; fila < alto; ++fila) {
    std::string const & linea = filas_texto[static_cast<std::size_t>(fila)];
    if (!linea.empty()) {
      // cert-err33-c: ignoramos explícitamente el valor devuelto
      (void) std::fwrite(linea.data(), 1, linea.size(), archivo.get());  // NOLINT(cert-err33-c)
    }
  }
  return true;
}
