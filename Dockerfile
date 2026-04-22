# Etapa 1: Compilación
FROM rust:latest as builder
WORKDIR /usr/src/guardian
COPY Cargo.toml .
COPY src ./src
# Compilar en modo release (máxima velocidad)
RUN cargo build --release

# Etapa 2: Imagen Final (Ubuntu Moderno)
FROM ubuntu:24.04
# Instalar dependencias necesarias para ONNX Runtime
RUN apt-get update && apt-get install -y libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copiar el ejecutable compilado
COPY --from=builder /usr/src/guardian/target/release/guardian_server /app/guardian_server
# Crear carpeta para modelos
RUN mkdir -p /app/models

EXPOSE 8080
CMD ./guardian_server