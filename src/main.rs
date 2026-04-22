mod ai_engine;

use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::ai_engine::AiEngine;
use base64::{Engine as _, engine::general_purpose}; // <- Nueva forma de usar base64

#[derive(Deserialize)]
struct ScanRequest {
    context: String, 
    data_base64: String,
}

#[derive(Serialize)]
struct ScanResponse {
    verdict: String,
    risk_score: f32,
    timestamp: String,
}

struct AppState {
    ai_engine: Arc<AiEngine>,
}

#[post("/api/v3/scan")]
async fn scan(
    req: web::Json<ScanRequest>,
    state: web::Data<AppState>,
) -> impl Responder {
    // 1. Decodificar Base64 (API Moderna)
    let bytes = match general_purpose::STANDARD.decode(&req.data_base64) {
        Ok(b) => b,
        Err(_) => return HttpResponse::BadRequest().json("Invalid base64 data"),
    };

    // 2. Analizar usando el motor de IA
    match state.ai_engine.analyze(&req.context, &bytes) {
        Ok(risk_score) => {
            let verdict = if risk_score > 0.45 { "MALICIOUS".to_string() } else { "BENIGN".to_string() };
            
            HttpResponse::Ok().json(ScanResponse {
                verdict,
                risk_score,
                timestamp: chrono::Utc::now().to_rfc3339(),
            })
        },
        Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    log::info!("🚀 Iniciando Guardian AI v3.0 Core...");

    let ai_engine = AiEngine::new("./models").expect("Error cargando modelos ONNX");
    let app_state = web::Data::new(AppState {
        ai_engine: Arc::new(ai_engine),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(scan)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}