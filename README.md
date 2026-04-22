Entendido, he eliminado las citas del texto final. Aquí tienes la versión limpia y profesional del `README.md` para tu repositorio de GitHub, basada fielmente en tu documentación maestra.

-----

# 🛡️ Guardian AI: Intelligent Security Engine

[](https://www.rust-lang.org)
[](https://www.pytorch.org)
[](https://onnx.ai)
[](https://www.docker.com)

**Guardian AI** es un servidor de análisis de seguridad de alto rendimiento escrito en **Rust**. Actúa como un motor de inteligencia artificial que analiza datos en tiempo real para determinar si son maliciosos o benignos, utilizando 6 modelos de red neuronal en formato **ONNX**.

## 🚀 Características Clave

  * **Alto Rendimiento:** Servidor web asíncrono construido con `Actix-Web` y `Tokio`.
  * **Inferencia Especializada:** Utiliza 6 modelos ONNX entrenados para vectores de ataque específicos.
  * **Arquitectura NeuroBrain:** Basada en redes neuronales recurrentes (**GRU Bidireccionales**) que capturan patrones secuenciales en los datos.
  * **Seguridad y Concurrencia:** Implementación de `Mutex` por sesión ONNX para garantizar hilos seguros en Rust.
  * **Despliegue Sencillo:** Totalmente dockerizado para una implementación consistente en cualquier entorno.

## 🧠 Modelos de Inteligencia Artificial

El sistema cuenta con "cerebros" entrenados con datasets reales para contextos específicos de ciberseguridad:

| Contexto | Dataset Origen | Función de Seguridad |
| :--- | :--- | :--- |
| **WAF** | CSIC 2010 | Web Application Firewall (SQLi, XSS, LFI) |
| **AV** | Malware Detection | Antivirus — patrones de malware en binarios |
| **IDS** | NSL-KDD | Detección de Intrusos y anomalías en red |
| **LOG** | LogHub (BGL) | Análisis de actividad sospechosa en registros |
| **DLP** | Enron Spam | Prevención de pérdida de datos sensibles |
| **HONEYPOT** | NSL-KDD | Clasificación de interacciones con trampas |

## 📐 Fundamentos Matemáticos

El proyecto no solo implementa código, sino que se basa en principios matemáticos de aprendizaje profundo:

### 1\. Preprocesamiento (Tensor de Bytes)

Todos los datos se normalizan a un vector fijo $x \in \{0, 1, \dots, 254\}^{1024}$. Para datos numéricos, se aplica la operación módulo:
$$valor\_byte = feature\_numérica \pmod{255}$$

### 2\. Arquitectura GRU Bidireccional

Cada celda GRU procesa la información mediante puertas de actualización ($z_t$) y reseteo ($r_t$):

  * **Update Gate:** $z_t = \sigma( W_z \cdot x_t + U_z \cdot h_{t-1} + b_z )$
  * **Reset Gate:** $r_t = \sigma( W_r \cdot x_t + U_r \cdot h_{t-1} + b_r )$

Al ser bidireccional, el sistema concatena los estados ocultos de ambas direcciones para capturar el contexto completo del payload:
$$o_t = [ h_t^{forward} ; h_t^{backward} ] \in \mathbb{R}^{64}$$

## 🛠️ Instalación y Uso

### Despliegue con Docker (Recomendado)

```bash
cd guardian/
docker-compose up --build
```

El servidor estará disponible en `http://localhost:8080`.

### Uso de la API REST

Envía un payload en Base64 al endpoint único:

  * **Método:** `POST`
  * **URL:** `/api/v3/scan`

**Ejemplo de Request:**

```json
{
  "context": "waf",
  "data_base64": "R0VUIC9hZG1pbi9wYW5lbA=="
}
```

## 📈 Hoja de Ruta y Mejoras

  * **Arquitecturas Avanzadas:** Integración de *Transformers* (Self-Attention) para dependencias de largo alcance.
  * **Rendimiento:** Implementación de un pool de sesiones ONNX y aceleración por GPU (CUDA).
  * **Observabilidad:** Integración de métricas con Prometheus y visualización en Grafana.

-----

**Desarrollado por:** Paul Mauricio Cano Amparo
