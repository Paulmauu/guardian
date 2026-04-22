use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex; // <-- Importamos el candado de seguridad

pub struct AiEngine {
    // Envolvemos cada sesión dentro de un Mutex
    models: HashMap<String, Mutex<Session>>,
}

impl AiEngine {
    pub fn new(models_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut models = HashMap::new();
        let contexts = vec!["waf", "av", "ids", "log", "dlp", "honeypot"];

        for ctx in contexts {
            let path = format!("{}/brain_{}.onnx", models_dir, ctx);
            if Path::new(&path).exists() {
                // Configura y carga el modelo ONNX en memoria
                let session = Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .commit_from_file(&path)?;
                
                // Guardamos el modelo protegido por el candado
                models.insert(ctx.to_string(), Mutex::new(session));
                log::info!("✅ Cerebro ONNX cargado: {}", ctx);
            } else {
                log::warn!("⚠️ No se encontró el modelo para: {}", ctx);
            }
        }
        Ok(Self { models })
    }

    pub fn analyze(&self, context: &str, raw_data: &[u8]) -> Result<f32, Box<dyn std::error::Error>> {
        let session_mutex = match self.models.get(context) {
            Some(s) => s,
            None => return Err(format!("Contexto '{}' no soportado o modelo no cargado", context).into()),
        };

        // 1. Preprocesar datos (Truncar o rellenar con ceros a 1024 bytes)
        let mut input_array = vec![0i64; 1024];
        let len = std::cmp::min(raw_data.len(), 1024);
        for i in 0..len {
            input_array[i] = raw_data[i] as i64;
        }

        // 2 & 3. Creamos el tensor pasando (Forma, Datos) directamente
        let input_value = Value::from_array(([1_usize, 1024_usize], input_array))?;

        // 4. Pedimos la llave del candado (bloqueo seguro para múltiples hilos)
        let mut session = session_mutex.lock().unwrap();

        // 5. Ejecutar Inferencia ONNX (Ahora sí nos deja usar mutabilidad)
        let outputs = session.run(inputs![input_value])?;

        // 6. Extraer Probabilidad
        let prob_tuple = outputs["probability"].try_extract_tensor::<f32>()?;
        let threat_prob = prob_tuple.1[1]; 

        Ok(threat_prob)
    }
}