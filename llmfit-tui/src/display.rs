use colored::*;
use llmfit_core::fit::{FitLevel, ModelFit};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::LlmModel;
use llmfit_core::plan::PlanEstimate;
use tabled::{Table, Tabled, settings::Style};

#[derive(Tabled)]
struct ModelRow {
    #[tabled(rename = "状态")]
    status: String,
    #[tabled(rename = "模型")]
    name: String,
    #[tabled(rename = "提供商")]
    provider: String,
    #[tabled(rename = "参数")]
    size: String,
    #[tabled(rename = "评分")]
    score: String,
    #[tabled(rename = "tok/s 预估")]
    tps: String,
    #[tabled(rename = "量化")]
    quant: String,
    #[tabled(rename = "运行时")]
    runtime: String,
    #[tabled(rename = "模式")]
    mode: String,
    #[tabled(rename = "内存 %")]
    mem_use: String,
    #[tabled(rename = "上下文")]
    context: String,
}

pub fn display_all_models(models: &[LlmModel]) {
    println!("\n{}", "=== 可用的大模型 ===".bold().cyan());
    println!("模型总数: {}\n", models.len());

    let rows: Vec<ModelRow> = models
        .iter()
        .map(|m| ModelRow {
            status: "--".to_string(),
            name: m.name.clone(),
            provider: m.provider.clone(),
            size: m.parameter_count.clone(),
            score: "-".to_string(),
            tps: "-".to_string(),
            quant: m.quantization.clone(),
            runtime: "-".to_string(),
            mode: "-".to_string(),
            mem_use: "-".to_string(),
            context: format!("{}k", m.context_length / 1000),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
}

pub fn display_model_fits(fits: &[ModelFit]) {
    if fits.is_empty() {
        println!("\n{}", "未找到兼容您系统的模型。".yellow());
        return;
    }

    println!("\n{}", "=== 模型兼容性分析 ===".bold().cyan());
    println!("找到 {} 个兼容的模型\n", fits.len());

    let rows: Vec<ModelRow> = fits
        .iter()
        .map(|fit| {
            let status_text = format!("{} {}", fit.fit_emoji(), fit.fit_text());

            ModelRow {
                status: status_text,
                name: fit.model.name.clone(),
                provider: fit.model.provider.clone(),
                size: fit.model.parameter_count.clone(),
                score: format!("{:.0}", fit.score),
                tps: format!("{:.1}", fit.estimated_tps),
                quant: fit.best_quant.clone(),
                runtime: fit.runtime_text().to_string(),
                mode: fit.run_mode_text().to_string(),
                mem_use: format!("{:.1}%", fit.utilization_pct),
                context: format!("{}k", fit.model.context_length / 1000),
            }
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
    println!("  注: tok/s 速度仅为基准预估；实际运行速度取决于推理引擎与系统负载。");
}

pub fn display_model_detail(fit: &ModelFit) {
    println!("\n{}", format!("=== {} ===", fit.model.name).bold().cyan());
    println!();
    println!("{}: {}", "提供商".bold(), fit.model.provider);
    println!("{}: {}", "参数规模".bold(), fit.model.parameter_count);
    println!("{}: {}", "推荐量化".bold(), fit.model.quantization);
    println!("{}: {}", "最佳量化".bold(), fit.best_quant);
    println!("{}: {} tokens", "上下文".bold(), fit.model.context_length);
    println!("{}: {}", "用途".bold(), fit.model.use_case);
    println!("{}: {}", "分类".bold(), fit.use_case.label());
    if let Some(ref date) = fit.model.release_date {
        println!("{}: {}", "发布时间".bold(), date);
    }
    println!(
        "{}: {} (基准预估 ~{:.1} tok/s)",
        "运行时".bold(),
        fit.runtime_text(),
        fit.estimated_tps
    );
    println!();

    println!("{}", "评分细则:".bold().underline());
    println!("  总分: {:.1} / 100", fit.score);
    println!(
        "  质量: {:.0}  速度: {:.0}  匹配度: {:.0}  上下文: {:.0}",
        fit.score_components.quality,
        fit.score_components.speed,
        fit.score_components.fit,
        fit.score_components.context
    );
    println!("  基准预估速度: {:.1} tok/s", fit.estimated_tps);
    println!();

    println!("{}", "资源需求:".bold().underline());
    if let Some(vram) = fit.model.min_vram_gb {
        println!("  最低显存: {:.1} GB", vram);
    }
    println!("  最低内存: {:.1} GB (仅 CPU 推理)", fit.model.min_ram_gb);
    println!("  推荐内存: {:.1} GB", fit.model.recommended_ram_gb);

    // MoE Architecture info
    if fit.model.is_moe {
        println!();
        println!("{}", "MoE 架构:".bold().underline());
        if let (Some(num_experts), Some(active_experts)) =
            (fit.model.num_experts, fit.model.active_experts)
        {
            println!(
                "  专家网络: {} 激活 / {} 总共 (每 token)",
                active_experts, num_experts
            );
        }
        if let Some(active_vram) = fit.model.moe_active_vram_gb() {
            println!(
                "  激活显存: {:.1} GB (对比 {:.1} GB 完整模型)",
                active_vram,
                fit.model.min_vram_gb.unwrap_or(0.0)
            );
        }
        if let Some(offloaded) = fit.moe_offloaded_gb {
            println!("  卸载内存: {:.1} GB 未激活专家存放于内存", offloaded);
        }
    }
    println!();

    println!("{}", "系统匹配分析:".bold().underline());

    let fit_color = match fit.fit_level {
        FitLevel::Perfect => "green",
        FitLevel::Good => "yellow",
        FitLevel::Marginal => "orange",
        FitLevel::TooTight => "red",
    };

    println!(
        "  状态: {} {}",
        fit.fit_emoji(),
        fit.fit_text().color(fit_color)
    );
    println!("  运行模式: {}", fit.run_mode_text());
    println!(
        "  内存利用率: {:.1}% ({:.1} / {:.1} GB)",
        fit.utilization_pct, fit.memory_required_gb, fit.memory_available_gb
    );
    println!();

    if !fit.model.gguf_sources.is_empty() {
        println!("{}", "GGUF 下载来源:".bold().underline());
        for src in &fit.model.gguf_sources {
            println!("  {} → https://huggingface.co/{}", src.provider, src.repo);
        }
        println!(
            "  {}",
            format!(
                "提示: llmfit download {} --quant {}",
                fit.model.gguf_sources[0].repo, fit.best_quant
            )
            .dimmed()
        );
        println!();
    }

    if !fit.notes.is_empty() {
        println!("{}", "备注:".bold().underline());
        for note in &fit.notes {
            println!("  {}", note);
        }
        println!();
    }
}

pub fn display_model_diff(fits: &[ModelFit], sort_label: &str) {
    if fits.len() < 2 {
        println!("\n{}", "需要至少2个模型进行对比。".yellow());
        return;
    }

    println!("\n{}", "=== 模型对比分析 ===".bold().cyan());
    println!("正在对比 {} 个模型 (按 {} 排序)\n", fits.len(), sort_label);

    let metric_width = 20usize;
    let col_width = 32usize;

    let model_headers: Vec<String> = fits
        .iter()
        .enumerate()
        .map(|(i, fit)| {
            let label = format!("M{}: {}", i + 1, fit.model.name);
            truncate_to_width(&label, col_width)
        })
        .collect();

    print!("{:<metric_width$}", "指标".bold());
    for header in &model_headers {
        print!("  {:<col_width$}", header.bold());
    }
    println!();

    print!("{:-<metric_width$}", "");
    for _ in &model_headers {
        print!("  {:-<col_width$}", "");
    }
    println!();

    let base = &fits[0];

    print_metric_row(
        "评分",
        fits.iter()
            .map(|f| format_with_delta(format!("{:.1}", f.score), f.score - base.score))
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "基准预估 tok/s",
        fits.iter()
            .map(|f| {
                format_with_delta(
                    format!("{:.1}", f.estimated_tps),
                    f.estimated_tps - base.estimated_tps,
                )
            })
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "匹配度",
        fits.iter()
            .map(|f| format!("{} {}", f.fit_emoji(), f.fit_text()))
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "运行模式",
        fits.iter().map(|f| f.run_mode_text().to_string()).collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "运行时",
        fits.iter().map(|f| f.runtime_text().to_string()).collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "内存 %",
        fits.iter()
            .map(|f| {
                format_with_delta(
                    format!("{:.1}%", f.utilization_pct),
                    f.utilization_pct - base.utilization_pct,
                )
            })
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "参数规模",
        fits.iter()
            .map(|f| f.model.parameter_count.clone())
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "上下文",
        fits.iter()
            .map(|f| format!("{} tokens", f.model.context_length))
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "最佳量化",
        fits.iter().map(|f| f.best_quant.clone()).collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "提供商",
        fits.iter().map(|f| f.model.provider.clone()).collect(),
        metric_width,
        col_width,
    );
}

fn print_metric_row(metric: &str, values: Vec<String>, metric_width: usize, col_width: usize) {
    print!("{:<metric_width$}", metric);
    for value in values {
        print!("  {:<col_width$}", truncate_to_width(&value, col_width));
    }
    println!();
}

fn format_with_delta(value: String, delta: f64) -> String {
    if delta.abs() < 0.05 {
        return value;
    }
    format!("{} ({:+.1})", value, delta)
}

fn truncate_to_width(input: &str, width: usize) -> String {
    if input.chars().count() <= width {
        return input.to_string();
    }
    let mut out = input
        .chars()
        .take(width.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

pub fn display_search_results(models: &[&LlmModel], query: &str) {
    if models.is_empty() {
        println!("\n{}", format!("未找到匹配 '{}' 的模型", query).yellow());
        return;
    }

    println!(
        "\n{}",
        format!("=== '{}' 的搜索结果 ===", query).bold().cyan()
    );
    println!("找到 {} 个模型\n", models.len());

    let rows: Vec<ModelRow> = models
        .iter()
        .map(|m| ModelRow {
            status: "--".to_string(),
            name: m.name.clone(),
            provider: m.provider.clone(),
            size: m.parameter_count.clone(),
            score: "-".to_string(),
            tps: "-".to_string(),
            quant: m.quantization.clone(),
            runtime: "-".to_string(),
            mode: "-".to_string(),
            mem_use: "-".to_string(),
            context: format!("{}k", m.context_length / 1000),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
}

// ────────────────────────────────────────────────────────────────────
// JSON output for machine consumption (OpenClaw skills, scripts, etc.)
// ────────────────────────────────────────────────────────────────────

/// Serialize system specs to JSON and print to stdout.
pub fn display_json_system(specs: &SystemSpecs) {
    let output = serde_json::json!({
        "system": system_json(specs),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

/// Serialize system specs + model fits to JSON and print to stdout.
pub fn display_json_fits(specs: &SystemSpecs, fits: &[ModelFit]) {
    let models: Vec<serde_json::Value> = fits.iter().map(fit_to_json).collect();
    let output = serde_json::json!({
        "system": system_json(specs),
        "models": models,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

/// Serialize diff output via serde derives (new diff-only path).
pub fn display_json_diff_fits(specs: &SystemSpecs, fits: &[ModelFit]) {
    #[derive(serde::Serialize)]
    struct FitsOutput<'a> {
        system: &'a SystemSpecs,
        models: &'a [ModelFit],
    }
    let output = FitsOutput {
        system: specs,
        models: fits,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

fn system_json(specs: &SystemSpecs) -> serde_json::Value {
    let gpus_json: Vec<serde_json::Value> = specs
        .gpus
        .iter()
        .map(|g| {
            serde_json::json!({
                "name": g.name,
                "vram_gb": g.vram_gb.map(round2),
                "backend": g.backend.label(),
                "count": g.count,
                "unified_memory": g.unified_memory,
            })
        })
        .collect();

    serde_json::json!({
        "total_ram_gb": round2(specs.total_ram_gb),
        "available_ram_gb": round2(specs.available_ram_gb),
        "cpu_cores": specs.total_cpu_cores,
        "cpu_name": specs.cpu_name,
        "has_gpu": specs.has_gpu,
        "gpu_vram_gb": specs.gpu_vram_gb.map(round2),
        "gpu_name": specs.gpu_name,
        "gpu_count": specs.gpu_count,
        "unified_memory": specs.unified_memory,
        "backend": specs.backend.label(),
        "gpus": gpus_json,
    })
}

fn fit_to_json(fit: &ModelFit) -> serde_json::Value {
    serde_json::json!({
        "name": fit.model.name,
        "provider": fit.model.provider,
        "parameter_count": fit.model.parameter_count,
        "params_b": round2(fit.model.params_b()),
        "context_length": fit.model.context_length,
        "use_case": fit.model.use_case,
        "category": fit.use_case.label(),
        "release_date": fit.model.release_date,
        "is_moe": fit.model.is_moe,
        "fit_level": fit.fit_text(),
        "run_mode": fit.run_mode_text(),
        "score": round1(fit.score),
        "score_components": {
            "quality": round1(fit.score_components.quality),
            "speed": round1(fit.score_components.speed),
            "fit": round1(fit.score_components.fit),
            "context": round1(fit.score_components.context),
        },
        "estimated_tps": round1(fit.estimated_tps),
        "runtime": fit.runtime_text(),
        "runtime_label": fit.runtime.label(),
        "best_quant": fit.best_quant,
        "memory_required_gb": round2(fit.memory_required_gb),
        "memory_available_gb": round2(fit.memory_available_gb),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": fit.notes,
        "gguf_sources": fit.model.gguf_sources,
    })
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub fn display_model_plan(plan: &PlanEstimate) {
    println!("\n{}", "=== 硬件规划预估 ===".bold().cyan());
    println!("{} {}", "模型:".bold(), plan.model_name);
    println!("{} {}", "提供商:".bold(), plan.provider);
    println!("{} {}", "上下文:".bold(), plan.context);
    println!("{} {}", "量化:".bold(), plan.quantization);
    if let Some(tps) = plan.target_tps {
        println!("{} {:.1} tok/s", "目标 TPS:".bold(), tps);
    }
    println!("{} {}", "备注:".bold(), plan.estimate_notice);
    println!();

    println!("{}", "最低硬件要求:".bold().underline());
    println!(
        "  显存: {}",
        plan.minimum
            .vram_gb
            .map(|v| format!("{v:.1} GB"))
            .unwrap_or_else(|| "无需配置".to_string())
    );
    println!("  内存: {:.1} GB", plan.minimum.ram_gb);
    println!("  CPU 核数: {}", plan.minimum.cpu_cores);
    println!();

    println!("{}", "推荐硬件要求:".bold().underline());
    println!(
        "  显存: {}",
        plan.recommended
            .vram_gb
            .map(|v| format!("{v:.1} GB"))
            .unwrap_or_else(|| "无需配置".to_string())
    );
    println!("  内存: {:.1} GB", plan.recommended.ram_gb);
    println!("  CPU 核数: {}", plan.recommended.cpu_cores);
    println!();

    println!("{}", "支持的运行路径:".bold().underline());
    for path in &plan.run_paths {
        println!(
            "  {}: {}",
            path.path.label(),
            if path.feasible { "支持" } else { "不支持" }
        );
        if let Some(min) = &path.minimum {
            println!(
                "    最低要求: 显存={} 内存={:.1} GB 核数={}",
                min.vram_gb
                    .map(|v| format!("{v:.1} GB"))
                    .unwrap_or_else(|| "无".to_string()),
                min.ram_gb,
                min.cpu_cores
            );
        }
        if let Some(tps) = path.estimated_tps {
            println!("    预估速度: {:.1} tok/s", tps);
        }
    }
    println!();

    println!("{}", "升级建议:".bold().underline());
    if plan.upgrade_deltas.is_empty() {
        println!("  为达到当前目标无需升级硬件。");
    } else {
        for delta in &plan.upgrade_deltas {
            println!("  {}", delta.description);
        }
    }
    println!();
}

pub fn display_json_plan(plan: &PlanEstimate) {
    println!(
        "{}",
        serde_json::to_string_pretty(plan).expect("JSON serialization failed")
    );
}
