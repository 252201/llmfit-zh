use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Table, TableState, Wrap,
    },
};

use crate::theme::ThemeColors;
use crate::tui_app::{
    App, AvailabilityFilter, DownloadCapability, DownloadProvider, FitFilter, InputMode, PlanField,
};
use llmfit_core::fit::{FitLevel, ModelFit, SortColumn};
use llmfit_core::hardware::is_running_in_wsl;
use llmfit_core::providers;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let tc = app.theme.colors();

    // Fill background if theme specifies one
    if tc.bg != Color::Reset {
        let bg_block = Block::default().style(Style::default().bg(tc.bg));
        frame.render_widget(bg_block, frame.area());
    }

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // system info bar
            Constraint::Length(3), // search + filters
            Constraint::Min(10),   // main table
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

    draw_system_bar(frame, app, outer[0], &tc);
    draw_search_and_filters(frame, app, outer[1], &tc);

    if app.show_plan {
        draw_plan(frame, app, outer[2], &tc);
    } else if app.show_multi_compare {
        draw_multi_compare(frame, app, outer[2], &tc);
    } else if app.show_compare {
        draw_compare(frame, app, outer[2], &tc);
    } else if app.show_detail {
        draw_detail(frame, app, outer[2], &tc);
    } else {
        draw_table(frame, app, outer[2], &tc);
    }

    draw_status_bar(frame, app, outer[3], &tc);

    // Draw popup overlays on top if active
    if app.input_mode == InputMode::ProviderPopup {
        draw_provider_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::UseCasePopup {
        draw_use_case_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::CapabilityPopup {
        draw_capability_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::DownloadProviderPopup {
        draw_download_provider_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::QuantPopup {
        draw_quant_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::RunModePopup {
        draw_run_mode_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::ParamsBucketPopup {
        draw_params_bucket_popup(frame, app, &tc);
    }
}

fn draw_system_bar(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let gpu_info = if app.specs.gpus.is_empty() {
        format!("GPU: none ({})", app.specs.backend.label())
    } else {
        let primary = &app.specs.gpus[0];
        let backend = primary.backend.label();
        let primary_str = if primary.unified_memory {
            format!(
                "{} ({:.1} GB shared, {})",
                primary.name,
                primary.vram_gb.unwrap_or(0.0),
                backend
            )
        } else {
            match primary.vram_gb {
                Some(vram) if vram > 0.0 => {
                    if primary.count > 1 {
                        let total_vram = vram * primary.count as f64;
                        format!(
                            "{} x{} ({:.1} GB each = {:.0} GB total, {})",
                            primary.name, primary.count, vram, total_vram, backend
                        )
                    } else {
                        format!("{} ({:.1} GB, {})", primary.name, vram, backend)
                    }
                }
                Some(_) => format!("{} (shared, {})", primary.name, backend),
                None => format!("{} ({})", primary.name, backend),
            }
        };
        let extra = app.specs.gpus.len() - 1;
        if extra > 0 {
            format!("GPU: {} +{} more", primary_str, extra)
        } else {
            format!("GPU: {}", primary_str)
        }
    };

    let ollama_info = if app.ollama_available {
        format!("Ollama: ✓ (已安装 {})", app.ollama_installed_count)
    } else {
        "Ollama: ✗".to_string()
    };
    let ollama_color = if app.ollama_available {
        tc.good
    } else {
        tc.muted
    };

    let mlx_info = if app.mlx_available {
        format!("MLX: ✓ (已安装 {})", app.mlx_installed.len())
    } else if !app.mlx_installed.is_empty() {
        format!("MLX: (已缓存 {})", app.mlx_installed.len())
    } else {
        "MLX: ✗".to_string()
    };
    let mlx_color = if app.mlx_available {
        tc.good
    } else if !app.mlx_installed.is_empty() {
        tc.warning
    } else {
        tc.muted
    };

    let llamacpp_info = if app.llamacpp_available {
        format!("llama.cpp: ✓ (模型数 {})", app.llamacpp_installed_count)
    } else if !app.llamacpp_installed.is_empty() {
        format!("llama.cpp: (已缓存 {})", app.llamacpp_installed_count)
    } else {
        "llama.cpp: ✗".to_string()
    };
    let llamacpp_color = if app.llamacpp_available {
        tc.good
    } else if !app.llamacpp_installed.is_empty() {
        tc.warning
    } else {
        tc.muted
    };

    let mut spans = vec![
        Span::styled(" CPU: ", Style::default().fg(tc.muted)),
        Span::styled(
            format!(
                "{} ({} cores)",
                app.specs.cpu_name, app.specs.total_cpu_cores
            ),
            Style::default().fg(tc.fg),
        ),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled("RAM: ", Style::default().fg(tc.muted)),
        Span::styled(
            format!(
                "{:.1} GB 可用 / {:.1} GB 总共{}",
                app.specs.available_ram_gb,
                app.specs.total_ram_gb,
                if is_running_in_wsl() { " (WSL)" } else { "" }
            ),
            Style::default().fg(tc.accent),
        ),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(gpu_info, Style::default().fg(tc.accent_secondary)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(ollama_info, Style::default().fg(ollama_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(mlx_info, Style::default().fg(mlx_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(llamacpp_info, Style::default().fg(llamacpp_color)),
    ];

    if app.backend_hidden_count > 0 {
        spans.push(Span::styled("  │  ", Style::default().fg(tc.muted)));
        spans.push(Span::styled(
            format!("已隐藏 {} 个模型 (不兼容的后端)", app.backend_hidden_count),
            Style::default().fg(tc.muted),
        ));
    }

    let text = Line::from(spans);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" llmfit ")
        .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));

    let paragraph = Paragraph::new(text).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_search_and_filters(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(30),    // search
            Constraint::Length(18), // provider summary
            Constraint::Length(18), // use-case summary
            Constraint::Length(16), // capability summary
            Constraint::Length(18), // sort column
            Constraint::Length(20), // fit filter
            Constraint::Length(20), // availability filter
            Constraint::Length(16), // theme
        ])
        .split(area);

    // Search box
    let search_style = match app.input_mode {
        InputMode::Search => Style::default().fg(tc.accent_secondary),
        InputMode::Normal
        | InputMode::Plan
        | InputMode::ProviderPopup
        | InputMode::UseCasePopup
        | InputMode::CapabilityPopup
        | InputMode::DownloadProviderPopup
        | InputMode::Visual
        | InputMode::Select
        | InputMode::QuantPopup
        | InputMode::RunModePopup
        | InputMode::ParamsBucketPopup => Style::default().fg(tc.muted),
    };

    let search_text = if app.search_query.is_empty() && app.input_mode == InputMode::Normal {
        Line::from(Span::styled("按 / 搜索...", Style::default().fg(tc.muted)))
    } else {
        Line::from(Span::styled(&app.search_query, Style::default().fg(tc.fg)))
    };

    let search_block = Block::default()
        .borders(Borders::ALL)
        .border_style(search_style)
        .title(" 搜索 ")
        .title_style(search_style);

    let search = Paragraph::new(search_text).block(search_block);
    frame.render_widget(search, chunks[0]);

    if app.input_mode == InputMode::Search {
        frame.set_cursor_position((
            chunks[0].x + app.cursor_position as u16 + 1,
            chunks[0].y + 1,
        ));
    }

    // Provider filter summary
    let active_count = app.selected_providers.iter().filter(|&&s| s).count();
    let total_count = app.providers.len();
    let provider_text = if active_count == total_count {
        "全部".to_string()
    } else {
        format!("{}/{}", active_count, total_count)
    };
    let provider_color = if active_count == total_count {
        tc.good
    } else if active_count == 0 {
        tc.error
    } else {
        tc.warning
    };

    let provider_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 提供商 (P) ")
        .title_style(Style::default().fg(tc.muted));

    let providers = Paragraph::new(Line::from(Span::styled(
        format!(" {}", provider_text),
        Style::default().fg(provider_color),
    )))
    .block(provider_block);
    frame.render_widget(providers, chunks[1]);

    // Use-case filter summary
    let active_count = app.selected_use_cases.iter().filter(|&&s| s).count();
    let total_count = app.use_cases.len();
    let use_case_text = if active_count == total_count {
        "全部".to_string()
    } else {
        format!("{}/{}", active_count, total_count)
    };
    let use_case_color = if active_count == total_count {
        tc.good
    } else if active_count == 0 {
        tc.error
    } else {
        tc.warning
    };

    let use_case_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 用途 (U) ")
        .title_style(Style::default().fg(tc.muted));

    let use_cases = Paragraph::new(Line::from(Span::styled(
        format!(" {}", use_case_text),
        Style::default().fg(use_case_color),
    )))
    .block(use_case_block);
    frame.render_widget(use_cases, chunks[2]);

    // Capability filter summary
    let active_cap_count = app.selected_capabilities.iter().filter(|&&s| s).count();
    let total_cap_count = app.capabilities.len();
    let cap_text = if active_cap_count == total_cap_count {
        "全部".to_string()
    } else {
        format!("{}/{}", active_cap_count, total_cap_count)
    };
    let cap_color = if active_cap_count == total_cap_count {
        tc.good
    } else if active_cap_count == 0 {
        tc.error
    } else {
        tc.warning
    };

    let cap_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 能力 (C) ")
        .title_style(Style::default().fg(tc.muted));

    let caps = Paragraph::new(Line::from(Span::styled(
        format!(" {}", cap_text),
        Style::default().fg(cap_color),
    )))
    .block(cap_block);
    frame.render_widget(caps, chunks[3]);

    // Sort column
    let sort_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 排序 [s] ")
        .title_style(Style::default().fg(tc.muted));

    let sort_text = Paragraph::new(Line::from(Span::styled(
        format!(" {}", app.sort_column.label()),
        Style::default().fg(tc.accent),
    )))
    .block(sort_block);
    frame.render_widget(sort_text, chunks[4]);

    // Fit filter
    let fit_style = match app.fit_filter {
        FitFilter::All => Style::default().fg(tc.fg),
        FitFilter::Runnable => Style::default().fg(tc.good),
        FitFilter::Perfect => Style::default().fg(tc.good),
        FitFilter::Good => Style::default().fg(tc.warning),
        FitFilter::Marginal => Style::default().fg(tc.fit_marginal),
        FitFilter::TooTight => Style::default().fg(tc.error),
    };

    let fit_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 匹配 [f] ")
        .title_style(Style::default().fg(tc.muted));

    let fit_text = Paragraph::new(Line::from(Span::styled(app.fit_filter.label(), fit_style)))
        .block(fit_block);
    frame.render_widget(fit_text, chunks[5]);

    // Availability filter
    let avail_style = match app.availability_filter {
        AvailabilityFilter::All => Style::default().fg(tc.fg),
        AvailabilityFilter::HasGguf => Style::default().fg(tc.info),
        AvailabilityFilter::Installed => Style::default().fg(tc.good),
    };

    let avail_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 可用 [a] ")
        .title_style(Style::default().fg(tc.muted));

    let avail_text = Paragraph::new(Line::from(Span::styled(
        app.availability_filter.label(),
        avail_style,
    )))
    .block(avail_block);
    frame.render_widget(avail_text, chunks[6]);

    // Theme indicator
    let theme_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" 主题 [t] ")
        .title_style(Style::default().fg(tc.muted));

    let theme_text = Paragraph::new(Line::from(Span::styled(
        format!(" {}", app.theme.label()),
        Style::default().fg(tc.info),
    )))
    .block(theme_block);
    frame.render_widget(theme_text, chunks[7]);
}

fn fit_color(level: FitLevel, tc: &ThemeColors) -> Color {
    match level {
        FitLevel::Perfect => tc.fit_perfect,
        FitLevel::Good => tc.fit_good,
        FitLevel::Marginal => tc.fit_marginal,
        FitLevel::TooTight => tc.fit_tight,
    }
}

fn fit_indicator(level: FitLevel) -> &'static str {
    match level {
        FitLevel::Perfect => "●",
        FitLevel::Good => "●",
        FitLevel::Marginal => "●",
        FitLevel::TooTight => "●",
    }
}

/// Build a compact animated download indicator for the "Inst" column.
fn pull_indicator(percent: Option<f64>, tick: u64) -> String {
    const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let spin = SPINNER[(tick as usize / 3) % SPINNER.len()];

    match percent {
        Some(pct) => {
            const BLOCKS: &[char] = &[' ', '░', '▒', '▓', '█'];
            let filled = pct / 100.0 * 3.0;
            let mut bar = String::with_capacity(5);
            bar.push(spin);
            for i in 0..3 {
                let level = (filled - i as f64).clamp(0.0, 1.0);
                let idx = (level * 4.0).round() as usize;
                bar.push(BLOCKS[idx]);
            }
            bar
        }
        None => format!(" {} ", spin),
    }
}

fn draw_table(frame: &mut Frame, app: &mut App, area: Rect, tc: &ThemeColors) {
    let sort_col = app.sort_column;
    let header_names = [
        "",
        "安装",
        "模型",
        "提供商",
        "参数",
        "评分",
        "tok/s*",
        "量化",
        "模式",
        "显存 %",
        "上下文",
        "日期",
        "匹配",
        "用途",
    ];
    let sort_col_idx: Option<usize> = match sort_col {
        SortColumn::Score => Some(5),
        SortColumn::Tps => Some(6),
        SortColumn::Params => Some(4),
        SortColumn::MemPct => Some(9),
        SortColumn::Ctx => Some(10),
        SortColumn::ReleaseDate => Some(11),
        SortColumn::UseCase => Some(13),
    };
    let in_select_mode = app.input_mode == InputMode::Select;
    let header_cells = header_names.iter().enumerate().map(|(i, h)| {
        if in_select_mode && app.select_column == i {
            Cell::from(format!("▸{}◂", h)).style(
                Style::default()
                    .fg(tc.fg)
                    .bg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            )
        } else if sort_col_idx == Some(i) {
            let arrow = if app.sort_ascending { "▲" } else { "▼" };
            Cell::from(format!("{} {}", h, arrow)).style(
                Style::default()
                    .fg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Cell::from(*h).style(Style::default().fg(tc.accent).add_modifier(Modifier::BOLD))
        }
    });
    let header = Row::new(header_cells).height(1);

    let visible_rows = (area.height as usize).saturating_sub(3).max(1);
    let total_rows = app.filtered_fits.len();
    let viewport_start = if total_rows <= visible_rows || app.selected_row < visible_rows {
        0
    } else {
        app.selected_row + 1 - visible_rows
    };
    let viewport_end = (viewport_start + visible_rows).min(total_rows);

    let visual_range = app.visual_range();
    let rows: Vec<Row> = app
        .filtered_fits
        .iter()
        .enumerate()
        .skip(viewport_start)
        .take(viewport_end.saturating_sub(viewport_start))
        .map(|(row_idx, &idx)| {
            let fit = &app.all_fits[idx];
            let color = fit_color(fit.fit_level, tc);

            let mode_color = match fit.run_mode {
                llmfit_core::fit::RunMode::Gpu => tc.mode_gpu,
                llmfit_core::fit::RunMode::MoeOffload => tc.mode_moe,
                llmfit_core::fit::RunMode::CpuOffload => tc.mode_offload,
                llmfit_core::fit::RunMode::CpuOnly => tc.mode_cpu,
            };

            let score_color = if fit.score >= 70.0 {
                tc.score_high
            } else if fit.score >= 50.0 {
                tc.score_mid
            } else {
                tc.score_low
            };

            #[allow(clippy::if_same_then_else)]
            let tps_text = if fit.estimated_tps >= 100.0 {
                format!("{:.0}", fit.estimated_tps)
            } else if fit.estimated_tps >= 10.0 {
                format!("{:.1}", fit.estimated_tps)
            } else {
                format!("{:.1}", fit.estimated_tps)
            };

            let is_pulling = app.pull_active.is_some()
                && app.pull_model_name.as_deref() == Some(&fit.model.name);
            let capability = app.download_capability_for(&fit.model.name);

            let installed_icon = if fit.installed {
                " ✓".to_string()
            } else if is_pulling {
                pull_indicator(app.pull_percent, app.tick_count)
            } else {
                match capability {
                    DownloadCapability::Unknown => " …".to_string(),
                    DownloadCapability::None => " —".to_string(),
                    DownloadCapability::Ollama => " O".to_string(),
                    DownloadCapability::LlamaCpp => " L".to_string(),
                    DownloadCapability::Both => "OL".to_string(),
                }
            };
            let installed_color = if fit.installed {
                tc.good
            } else if is_pulling {
                tc.warning
            } else {
                match capability {
                    DownloadCapability::Unknown => tc.muted,
                    DownloadCapability::None => tc.muted,
                    DownloadCapability::Ollama
                    | DownloadCapability::LlamaCpp
                    | DownloadCapability::Both => tc.info,
                }
            };

            let in_visual_range = visual_range
                .as_ref()
                .map(|r| r.contains(&row_idx))
                .unwrap_or(false);
            let row_style = if is_pulling {
                Style::default().bg(Color::Rgb(50, 50, 0))
            } else if in_visual_range {
                Style::default().bg(Color::Rgb(40, 40, 80))
            } else {
                Style::default()
            };

            let marker = if app.compare_mark_model.as_deref() == Some(fit.model.name.as_str()) {
                format!("{}*", fit_indicator(fit.fit_level))
            } else {
                fit_indicator(fit.fit_level).to_string()
            };

            Row::new(vec![
                Cell::from(marker).style(Style::default().fg(color)),
                Cell::from(installed_icon).style(Style::default().fg(installed_color)),
                Cell::from(fit.model.name.clone()).style(Style::default().fg(tc.fg)),
                Cell::from(fit.model.provider.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(fit.model.parameter_count.clone()).style(Style::default().fg(tc.fg)),
                Cell::from(format!("{:.0}", fit.score)).style(Style::default().fg(score_color)),
                Cell::from(tps_text).style(Style::default().fg(tc.fg)),
                Cell::from(fit.best_quant.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(fit.run_mode_text().to_string()).style(Style::default().fg(mode_color)),
                Cell::from(format!("{:.0}%", fit.utilization_pct))
                    .style(Style::default().fg(color)),
                Cell::from(format!("{}k", fit.model.context_length / 1000))
                    .style(Style::default().fg(tc.muted)),
                Cell::from(
                    fit.model
                        .release_date
                        .as_deref()
                        .and_then(|d| d.get(..7))
                        .unwrap_or("\u{2014}")
                        .to_string(),
                )
                .style(Style::default().fg(tc.muted)),
                Cell::from(fit.fit_text().to_string()).style(Style::default().fg(color)),
                Cell::from(fit.use_case.label().to_string()).style(Style::default().fg(tc.muted)),
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Length(2),  // indicator
        Constraint::Length(5),  // installed / pull %
        Constraint::Min(20),    // model name
        Constraint::Length(12), // provider
        Constraint::Length(8),  // params
        Constraint::Length(6),  // score
        Constraint::Length(6),  // tok/s
        Constraint::Length(10), // quant (AWQ-4bit, GPTQ-Int4, GPTQ-Int8)
        Constraint::Length(7),  // mode
        Constraint::Length(6),  // mem %
        Constraint::Length(5),  // ctx
        Constraint::Length(8),  // date (YYYY-MM)
        Constraint::Length(10), // fit
        Constraint::Min(10),    // use case
    ];

    let count_text = format!(
        " 模型 ({}/{}) ",
        app.filtered_fits.len(),
        app.all_fits.len()
    );

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(tc.border))
                .title(count_text)
                .title_style(Style::default().fg(tc.fg)),
        )
        .row_highlight_style(
            Style::default()
                .bg(tc.highlight_bg)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    let mut state = TableState::default();
    if !app.filtered_fits.is_empty() {
        state.select(Some(app.selected_row.saturating_sub(viewport_start)));
    }

    frame.render_stateful_widget(table, area, &mut state);

    // Scrollbar
    if app.filtered_fits.len() > (area.height as usize).saturating_sub(3) {
        let mut scrollbar_state =
            ScrollbarState::new(app.filtered_fits.len()).position(app.selected_row);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓")),
            area,
            &mut scrollbar_state,
        );
    }
}

fn draw_compare(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let Some((left, right)) = app.selected_compare_pair() else {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" 对比 ")
            .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));
        let body = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  对比功能需要选择两个不同的模型。",
                Style::default().fg(tc.warning),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  1) 移动到一个模型上并按下 m (标记)。",
                Style::default().fg(tc.muted),
            )),
            Line::from(Span::styled(
                "  2) 移动到另一个模型上并按下 c (对比)。",
                Style::default().fg(tc.muted),
            )),
            Line::from(Span::styled(
                "  3) 再次按下 c 返回。",
                Style::default().fg(tc.muted),
            )),
        ])
        .block(block);
        frame.render_widget(body, area);
        return;
    };

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(area);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(sections[1]);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" 对比 ", Style::default().fg(tc.accent).bold()),
        Span::styled(
            format!("{}  vs  {}", left.model.name, right.model.name),
            Style::default().fg(tc.fg),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border)),
    );
    frame.render_widget(title, sections[0]);

    let score_delta = right.score - left.score;
    let tps_delta = right.estimated_tps - left.estimated_tps;
    let mem_delta = right.utilization_pct - left.utilization_pct;
    let params_delta = right.model.params_b() - left.model.params_b();
    let ctx_delta = right.model.context_length as i64 - left.model.context_length as i64;

    let score_hint = if score_delta > 0.05 {
        " ↑"
    } else if score_delta < -0.05 {
        " ↓"
    } else {
        " ="
    };
    let tps_hint = if tps_delta > 0.05 {
        " ↑"
    } else if tps_delta < -0.05 {
        " ↓"
    } else {
        " ="
    };
    let mem_hint = if mem_delta > 0.05 {
        " ↑"
    } else if mem_delta < -0.05 {
        " ↓"
    } else {
        " ="
    };
    let params_hint = if params_delta > 0.01 {
        " ↑"
    } else if params_delta < -0.01 {
        " ↓"
    } else {
        " ="
    };
    let ctx_hint = if ctx_delta > 0 {
        " ↑"
    } else if ctx_delta < 0 {
        " ↓"
    } else {
        " ="
    };

    let score_style = Style::default().fg(if score_delta >= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let tps_style = Style::default().fg(if tps_delta >= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let mem_style = Style::default().fg(if mem_delta <= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let params_style = Style::default().fg(if params_delta >= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let ctx_style = Style::default().fg(if ctx_delta >= 0 { tc.good } else { tc.warning });

    let legend = Paragraph::new(Line::from(Span::styled(
        "  差异提示: ↑ 数值增加, ↓ 数值减少 (对于显存%, 越低越好)",
        Style::default().fg(tc.muted),
    )));
    frame.render_widget(legend, sections[0]);

    let left_metrics = CompareMetrics {
        score: format!("{:.1}", left.score),
        score_style: Style::default().fg(tc.score_high),
        tps: format!("{:.1}", left.estimated_tps),
        tps_style: Style::default().fg(tc.fg),
        mem: format!("{:.1}%", left.utilization_pct),
        mem_style: Style::default().fg(fit_color(left.fit_level, tc)),
        params: left.model.parameter_count.clone(),
        params_style: Style::default().fg(tc.fg),
        context: format!(" {} tokens", left.model.context_length),
        context_style: Style::default().fg(tc.fg),
    };

    let right_metrics = CompareMetrics {
        score: format!("{:.1} ({:+.1}){}", right.score, score_delta, score_hint),
        score_style,
        tps: format!("{:.1} ({:+.1}){}", right.estimated_tps, tps_delta, tps_hint),
        tps_style,
        mem: format!(
            "{:.1}% ({:+.1}%){}",
            right.utilization_pct, mem_delta, mem_hint
        ),
        mem_style,
        params: format!(
            "{} ({:+.2}B){}",
            right.model.parameter_count, params_delta, params_hint
        ),
        params_style,
        context: format!(
            " {} tokens ({:+}){}",
            right.model.context_length, ctx_delta, ctx_hint
        ),
        context_style: ctx_style,
    };

    render_compare_panel(frame, cols[0], tc, " 已标记 (基准) ", left, &left_metrics);
    render_compare_panel(
        frame,
        cols[1],
        tc,
        " 当前选中 (对比基准差异) ",
        right,
        &right_metrics,
    );
}

struct CompareMetrics {
    score: String,
    score_style: Style,
    tps: String,
    tps_style: Style,
    mem: String,
    mem_style: Style,
    params: String,
    params_style: Style,
    context: String,
    context_style: Style,
}

fn compare_badges(fit: &ModelFit) -> String {
    let mut tags = Vec::new();
    if fit.model.is_moe {
        tags.push("MoE");
    }
    if fit.run_mode == llmfit_core::fit::RunMode::MoeOffload {
        tags.push("Offload");
    }
    if !fit.notes.is_empty() {
        tags.push("Notes");
    }
    if tags.is_empty() {
        "-".to_string()
    } else {
        tags.join(", ")
    }
}

fn render_compare_panel(
    frame: &mut Frame,
    area: Rect,
    tc: &ThemeColors,
    title: &str,
    fit: &ModelFit,
    metrics: &CompareMetrics,
) {
    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  模型:   ", Style::default().fg(tc.muted)),
            Span::styled(fit.model.name.clone(), Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(vec![
            Span::styled("  提供商: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.model.provider),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  用途:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.use_case.label()),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  发布:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.model.release_date.as_deref().unwrap_or("未知")),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  评分:   ", Style::default().fg(tc.muted)),
            Span::styled(metrics.score.clone(), metrics.score_style),
        ]),
        Line::from(vec![
            Span::styled("  匹配:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} {}", fit_indicator(fit.fit_level), fit.fit_text()),
                Style::default().fg(fit_color(fit.fit_level, tc)),
            ),
        ]),
        Line::from(vec![
            Span::styled("  tok/s:  ", Style::default().fg(tc.muted)),
            Span::styled(metrics.tps.clone(), metrics.tps_style),
        ]),
        Line::from(vec![
            Span::styled("  显存%:  ", Style::default().fg(tc.muted)),
            Span::styled(metrics.mem.clone(), metrics.mem_style),
        ]),
        Line::from(vec![
            Span::styled("  运行时: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.runtime_text()),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  模式:   ", Style::default().fg(tc.muted)),
            Span::styled(fit.run_mode_text(), Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  参数:   ", Style::default().fg(tc.muted)),
            Span::styled(metrics.params.clone(), metrics.params_style),
        ]),
        Line::from(vec![
            Span::styled("  上下文: ", Style::default().fg(tc.muted)),
            Span::styled(metrics.context.clone(), metrics.context_style),
        ]),
        Line::from(vec![
            Span::styled("  量化:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} (默认 {})", fit.best_quant, fit.model.quantization),
                Style::default().fg(tc.good),
            ),
        ]),
        Line::from(vec![
            Span::styled("  标签:   ", Style::default().fg(tc.muted)),
            Span::styled(compare_badges(fit), Style::default().fg(tc.info)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(tc.border))
                .title(title)
                .title_style(Style::default().fg(tc.accent_secondary)),
        ),
        area,
    );
}

fn draw_multi_compare(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    if app.compare_models.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" 对比 ")
            .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));
        let body = Paragraph::new("  未选择用于对比的模型。").block(block);
        frame.render_widget(body, area);
        return;
    }

    let models: Vec<&ModelFit> = app
        .compare_models
        .iter()
        .filter_map(|&idx| app.all_fits.get(idx))
        .collect();

    if models.len() < 2 {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" 对比 ")
            .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));
        let body = Paragraph::new("  至少需要 2 个模型才能对比。").block(block);
        frame.render_widget(body, area);
        return;
    }

    // Attribute rows: label, value extractor, color logic
    struct AttrRow {
        label: &'static str,
        values: Vec<String>,
        styles: Vec<Style>,
    }

    let label_width: u16 = 12;
    // How many model columns can we fit?
    let available_width = area.width.saturating_sub(label_width + 3); // borders + label col
    let col_width: u16 = 20;
    let max_visible = (available_width / col_width).max(1) as usize;
    let scroll = app
        .compare_scroll
        .min(models.len().saturating_sub(max_visible));
    let visible_models: Vec<&ModelFit> = models
        .iter()
        .skip(scroll)
        .take(max_visible)
        .copied()
        .collect();
    let n = visible_models.len();

    // Find best/worst for highlighting
    let best_score = models.iter().map(|m| m.score).fold(f64::MIN, f64::max);
    let best_tps = models
        .iter()
        .map(|m| m.estimated_tps)
        .fold(f64::MIN, f64::max);
    let best_mem = models
        .iter()
        .map(|m| m.utilization_pct)
        .fold(f64::MAX, f64::min); // lower is better
    let best_ctx = models
        .iter()
        .map(|m| m.model.context_length)
        .max()
        .unwrap_or(0);

    let mut rows: Vec<AttrRow> = Vec::new();

    // Model name
    rows.push(AttrRow {
        label: "Model",
        values: visible_models
            .iter()
            .map(|m| truncate_str(&m.model.name, col_width as usize - 1))
            .collect(),
        styles: vec![Style::default().fg(tc.fg).add_modifier(Modifier::BOLD); n],
    });

    // Provider
    rows.push(AttrRow {
        label: "Provider",
        values: visible_models
            .iter()
            .map(|m| m.model.provider.clone())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Score
    rows.push(AttrRow {
        label: "Score",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1}", m.score))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if (m.score - best_score).abs() < 0.1 {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else if m.score >= 70.0 {
                    Style::default().fg(tc.score_high)
                } else if m.score >= 50.0 {
                    Style::default().fg(tc.score_mid)
                } else {
                    Style::default().fg(tc.score_low)
                }
            })
            .collect(),
    });

    // tok/s
    rows.push(AttrRow {
        label: "tok/s",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1}", m.estimated_tps))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if (m.estimated_tps - best_tps).abs() < 0.1 {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(tc.fg)
                }
            })
            .collect(),
    });

    // Fit
    rows.push(AttrRow {
        label: "Fit",
        values: visible_models
            .iter()
            .map(|m| format!("{} {}", fit_indicator(m.fit_level), m.fit_text()))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| Style::default().fg(fit_color(m.fit_level, tc)))
            .collect(),
    });

    // Mem %
    rows.push(AttrRow {
        label: "Mem %",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1}%", m.utilization_pct))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if (m.utilization_pct - best_mem).abs() < 0.1 {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(fit_color(m.fit_level, tc))
                }
            })
            .collect(),
    });

    // Params
    rows.push(AttrRow {
        label: "Params",
        values: visible_models
            .iter()
            .map(|m| m.model.parameter_count.clone())
            .collect(),
        styles: vec![Style::default().fg(tc.fg); n],
    });

    // Mode
    rows.push(AttrRow {
        label: "Mode",
        values: visible_models
            .iter()
            .map(|m| m.run_mode_text().to_string())
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                let c = match m.run_mode {
                    llmfit_core::fit::RunMode::Gpu => tc.mode_gpu,
                    llmfit_core::fit::RunMode::MoeOffload => tc.mode_moe,
                    llmfit_core::fit::RunMode::CpuOffload => tc.mode_offload,
                    llmfit_core::fit::RunMode::CpuOnly => tc.mode_cpu,
                };
                Style::default().fg(c)
            })
            .collect(),
    });

    // Context
    rows.push(AttrRow {
        label: "Context",
        values: visible_models
            .iter()
            .map(|m| format!("{}k", m.model.context_length / 1000))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if m.model.context_length == best_ctx {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(tc.muted)
                }
            })
            .collect(),
    });

    // Quant
    rows.push(AttrRow {
        label: "Quant",
        values: visible_models
            .iter()
            .map(|m| m.best_quant.clone())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Use Case
    rows.push(AttrRow {
        label: "Use Case",
        values: visible_models
            .iter()
            .map(|m| m.use_case.label().to_string())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Runtime
    rows.push(AttrRow {
        label: "Runtime",
        values: visible_models
            .iter()
            .map(|m| m.runtime_text().to_string())
            .collect(),
        styles: vec![Style::default().fg(tc.fg); n],
    });

    // Build the table
    let mut header_cells = vec![Cell::from("").style(Style::default().fg(tc.accent).bold())];
    for (i, m) in visible_models.iter().enumerate() {
        let name = truncate_str(&m.model.name, col_width as usize - 1);
        let style = if i == 0 && scroll == 0 {
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(tc.accent).add_modifier(Modifier::BOLD)
        };
        header_cells.push(Cell::from(name).style(style));
    }
    let header = Row::new(header_cells).height(1);

    let table_rows: Vec<Row> = rows
        .iter()
        .enumerate()
        .map(|(row_idx, attr)| {
            let mut cells =
                vec![Cell::from(attr.label).style(Style::default().fg(tc.muted).bold())];
            for (col_idx, (val, style)) in attr.values.iter().zip(attr.styles.iter()).enumerate() {
                let _ = col_idx;
                cells.push(Cell::from(val.as_str()).style(*style));
            }
            let bg = if row_idx % 2 == 0 {
                Style::default()
            } else {
                Style::default().bg(Color::Rgb(25, 25, 35))
            };
            Row::new(cells).style(bg)
        })
        .collect();

    let mut widths = vec![Constraint::Length(label_width)];
    for _ in 0..n {
        widths.push(Constraint::Length(col_width));
    }

    let scroll_info = if models.len() > max_visible {
        format!(" 对比 ({}/{})  ←/→ 滚动 ", models.len(), models.len())
    } else {
        format!(" 对比 ({} 个模型) ", models.len())
    };

    let table = Table::new(table_rows, widths).header(header).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(scroll_info)
            .title_style(
                Style::default()
                    .fg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            ),
    );

    frame.render_widget(table, area);
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}~", &s[..max_len.saturating_sub(1)])
    }
}

fn draw_detail(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let fit = match app.selected_fit() {
        Some(f) => f,
        None => {
            let block = Block::default().borders(Borders::ALL).title(" 未选择模型 ");
            frame.render_widget(block, area);
            return;
        }
    };

    let color = fit_color(fit.fit_level, tc);

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  模型:        ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.name, Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(vec![
            Span::styled("  提供商:      ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.provider, Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  参数:        ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.parameter_count, Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  推荐量化:    ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.model.quantization),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  最佳量化:    ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {} (基于当前硬件)", fit.best_quant),
                Style::default().fg(tc.good),
            ),
        ]),
        Line::from(vec![
            Span::styled("  上下文:      ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} tokens", fit.model.context_length),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  用途:        ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.use_case, Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  分类:        ", Style::default().fg(tc.muted)),
            Span::styled(fit.use_case.label(), Style::default().fg(tc.accent)),
        ]),
        Line::from(vec![
            Span::styled("  能力:        ", Style::default().fg(tc.muted)),
            Span::styled(
                if fit.model.capabilities.is_empty() {
                    " 无".to_string()
                } else {
                    format!(
                        " {}",
                        fit.model
                            .capabilities
                            .iter()
                            .map(|c| c.label())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                },
                Style::default().fg(tc.info),
            ),
        ]),
        Line::from(vec![
            Span::styled("  发布:        ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.model.release_date.as_deref().unwrap_or("未知"),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  运行时:      ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.runtime_text(),
                Style::default().fg(match fit.runtime {
                    llmfit_core::fit::InferenceRuntime::Mlx => tc.accent,
                    llmfit_core::fit::InferenceRuntime::Vllm => tc.accent_secondary,
                    _ => tc.fg,
                }),
            ),
            Span::styled(
                format!(" (基准预估 ~{:.1} tok/s)", fit.estimated_tps),
                Style::default().fg(tc.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  已安装:      ", Style::default().fg(tc.muted)),
            {
                let ollama_installed =
                    providers::is_model_installed(&fit.model.name, &app.ollama_installed);
                let mlx_installed =
                    providers::is_model_installed_mlx(&fit.model.name, &app.mlx_installed);
                let llamacpp_installed = providers::is_model_installed_llamacpp(
                    &fit.model.name,
                    &app.llamacpp_installed,
                );
                let any_available =
                    app.ollama_available || app.mlx_available || app.llamacpp_available;

                if ollama_installed && mlx_installed && llamacpp_installed {
                    Span::styled(
                        "✓ Ollama  ✓ MLX  ✓ llama.cpp",
                        Style::default().fg(tc.good).bold(),
                    )
                } else if ollama_installed && mlx_installed {
                    Span::styled("✓ Ollama  ✓ MLX", Style::default().fg(tc.good).bold())
                } else if ollama_installed && llamacpp_installed {
                    Span::styled("✓ Ollama  ✓ llama.cpp", Style::default().fg(tc.good).bold())
                } else if mlx_installed && llamacpp_installed {
                    Span::styled("✓ MLX  ✓ llama.cpp", Style::default().fg(tc.good).bold())
                } else if ollama_installed {
                    Span::styled("✓ Ollama", Style::default().fg(tc.good).bold())
                } else if mlx_installed {
                    Span::styled("✓ MLX", Style::default().fg(tc.good).bold())
                } else if llamacpp_installed {
                    Span::styled("✓ llama.cpp", Style::default().fg(tc.good).bold())
                } else if any_available {
                    Span::styled("✗ 否  (按 d 下载)", Style::default().fg(tc.muted))
                } else {
                    Span::styled("- 未检测到运行时", Style::default().fg(tc.muted))
                }
            },
        ]),
    ];

    // Scoring section
    let score_color = if fit.score >= 70.0 {
        tc.score_high
    } else if fit.score >= 50.0 {
        tc.score_mid
    } else {
        tc.score_low
    };
    lines.extend_from_slice(&[
        Line::from(""),
        Line::from(Span::styled(
            "  ── 评分细则 ──",
            Style::default().fg(tc.accent),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  总分:        ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} / 100", fit.score),
                Style::default().fg(score_color).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  质量:        ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.quality),
                Style::default().fg(tc.fg),
            ),
            Span::styled("  速度: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.speed),
                Style::default().fg(tc.fg),
            ),
            Span::styled("  匹配: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.fit),
                Style::default().fg(tc.fg),
            ),
            Span::styled("  上下文: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.context),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  基准预估:    ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} tok/s", fit.estimated_tps),
                Style::default().fg(tc.fg),
            ),
        ]),
    ]);

    // MoE Architecture section
    if fit.model.is_moe {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ── MoE 架构 ──",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(""));

        if let (Some(num_experts), Some(active_experts)) =
            (fit.model.num_experts, fit.model.active_experts)
        {
            lines.push(Line::from(vec![
                Span::styled("  专家网络:    ", Style::default().fg(tc.muted)),
                Span::styled(
                    format!("{} 激活 / {} 总共 (每 token)", active_experts, num_experts),
                    Style::default().fg(tc.accent),
                ),
            ]));
        }

        if let Some(active_vram) = fit.model.moe_active_vram_gb() {
            lines.push(Line::from(vec![
                Span::styled("  激活显存:    ", Style::default().fg(tc.muted)),
                Span::styled(
                    format!("{:.1} GB", active_vram),
                    Style::default().fg(tc.accent),
                ),
                Span::styled(
                    format!(
                        "  (对比 {:.1} GB 完整模型)",
                        fit.model.min_vram_gb.unwrap_or(0.0)
                    ),
                    Style::default().fg(tc.muted),
                ),
            ]));
        }

        if let Some(offloaded) = fit.moe_offloaded_gb {
            lines.push(Line::from(vec![
                Span::styled("  卸载内存:    ", Style::default().fg(tc.muted)),
                Span::styled(
                    format!("{:.1} GB 未激活专家存放于内存", offloaded),
                    Style::default().fg(tc.warning),
                ),
            ]));
        }

        if fit.run_mode == llmfit_core::fit::RunMode::MoeOffload {
            lines.push(Line::from(vec![
                Span::styled("  执行策略:    ", Style::default().fg(tc.muted)),
                Span::styled(
                    "专家卸载 (激活的在显存，未激活的在内存)",
                    Style::default().fg(tc.good),
                ),
            ]));
        } else if fit.run_mode == llmfit_core::fit::RunMode::Gpu {
            lines.push(Line::from(vec![
                Span::styled("  执行策略:    ", Style::default().fg(tc.muted)),
                Span::styled("所有专家加载到显存 (最优)", Style::default().fg(tc.good)),
            ]));
        }
    }

    lines.extend_from_slice(&[
        Line::from(""),
        Line::from(Span::styled(
            "  ── 系统匹配 ──",
            Style::default().fg(tc.accent),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  匹配程度:    ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} {}", fit_indicator(fit.fit_level), fit.fit_text()),
                Style::default().fg(color).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  运行模式:    ", Style::default().fg(tc.muted)),
            Span::styled(fit.run_mode_text(), Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  -- 内存信息 --",
            Style::default().fg(tc.accent),
        )),
        Line::from(""),
    ]);

    if let Some(vram) = fit.model.min_vram_gb {
        let vram_label = if app.specs.has_gpu {
            if app.specs.unified_memory {
                if let Some(sys_vram) = app.specs.gpu_vram_gb {
                    format!("  (shared: {:.1} GB)", sys_vram)
                } else {
                    "  (shared memory)".to_string()
                }
            } else if let Some(sys_vram) = app.specs.gpu_vram_gb {
                format!("  (system: {:.1} GB)", sys_vram)
            } else {
                "  (system: unknown)".to_string()
            }
        } else {
            "  (no GPU)".to_string()
        };
        lines.push(Line::from(vec![
            Span::styled("  Min VRAM:    ", Style::default().fg(tc.muted)),
            Span::styled(format!("{:.1} GB", vram), Style::default().fg(tc.fg)),
            Span::styled(vram_label, Style::default().fg(tc.muted)),
        ]));
    }

    lines.extend_from_slice(&[
        Line::from(vec![
            Span::styled("  Min RAM:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", fit.model.min_ram_gb),
                Style::default().fg(tc.fg),
            ),
            Span::styled(
                format!("  (system: {:.1} GB avail)", app.specs.available_ram_gb),
                Style::default().fg(tc.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Rec RAM:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", fit.model.recommended_ram_gb),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Mem Usage:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1}%", fit.utilization_pct),
                Style::default().fg(color),
            ),
            Span::styled(
                format!(
                    "  ({:.1} / {:.1} GB)",
                    fit.memory_required_gb, fit.memory_available_gb
                ),
                Style::default().fg(tc.muted),
            ),
        ]),
    ]);

    // Build right-pane content (GGUF sources + notes)
    let has_right_pane = !fit.model.gguf_sources.is_empty() || !fit.notes.is_empty();

    let mut right_lines: Vec<Line> = vec![Line::from("")];

    if !fit.model.gguf_sources.is_empty() {
        right_lines.push(Line::from(Span::styled(
            "  ── GGUF 下载 ──",
            Style::default().fg(tc.accent),
        )));
        right_lines.push(Line::from(""));
        for src in &fit.model.gguf_sources {
            right_lines.push(Line::from(vec![
                Span::styled(
                    format!("  📦 {:<12}", src.provider),
                    Style::default().fg(tc.info),
                ),
                Span::styled(format!("hf.co/{}", src.repo), Style::default().fg(tc.fg)),
            ]));
        }
        right_lines.push(Line::from(""));
        right_lines.push(Line::from(Span::styled(
            format!("  llmfit download \\"),
            Style::default().fg(tc.muted),
        )));
        right_lines.push(Line::from(Span::styled(
            format!("    {} \\", fit.model.gguf_sources[0].repo),
            Style::default().fg(tc.muted),
        )));
        right_lines.push(Line::from(Span::styled(
            format!("    --quant {}", fit.best_quant),
            Style::default().fg(tc.muted),
        )));
        right_lines.push(Line::from(""));
    }

    if !fit.notes.is_empty() {
        right_lines.push(Line::from(Span::styled(
            "  ── 备注 ──",
            Style::default().fg(tc.accent),
        )));
        right_lines.push(Line::from(""));
        for note in &fit.notes {
            right_lines.push(Line::from(Span::styled(
                format!("  {}", note),
                Style::default().fg(tc.fg),
            )));
        }
    }

    // Track the left pane area for cursor positioning
    let left_area;

    if has_right_pane {
        // Split into left (model info) and right (downloads + notes) panes
        let h_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);

        left_area = h_layout[0];

        let left_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(format!(" {} ", fit.model.name))
            .title_style(Style::default().fg(tc.fg).bold());

        let left_paragraph = Paragraph::new(lines)
            .block(left_block)
            .wrap(Wrap { trim: false });
        frame.render_widget(left_paragraph, h_layout[0]);

        let right_title = if !fit.model.gguf_sources.is_empty() {
            " 📦 下载与备注 "
        } else {
            " 备注 "
        };
        let right_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(right_title)
            .title_style(Style::default().fg(tc.info).bold());

        let right_paragraph = Paragraph::new(right_lines)
            .block(right_block)
            .wrap(Wrap { trim: false });
        frame.render_widget(right_paragraph, h_layout[1]);
    } else {
        left_area = area;

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(format!(" {} ", fit.model.name))
            .title_style(Style::default().fg(tc.fg).bold());

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false });
        frame.render_widget(paragraph, area);
    }

    if app.input_mode == InputMode::Plan {
        let (row_offset, label_len) = match app.plan_field {
            PlanField::Context => (5u16, 14),
            PlanField::Quant => (6u16, 14),
            PlanField::TargetTps => (7u16, 14),
        };
        let x = left_area.x + 1 + label_len + app.plan_cursor_position as u16;
        let y = left_area.y + 1 + row_offset;
        if x < left_area.x + left_area.width.saturating_sub(1)
            && y < left_area.y + left_area.height.saturating_sub(1)
        {
            frame.set_cursor_position((x, y));
        }
    }
}

fn draw_plan(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let Some(model_name) = app.plan_model_name() else {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" 规划器 ");
        frame.render_widget(block, area);
        return;
    };

    let field_style = |field: PlanField| {
        if app.input_mode == InputMode::Plan && app.plan_field == field {
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(tc.fg)
        }
    };

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  模型:       ", Style::default().fg(tc.muted)),
            Span::styled(model_name, Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(vec![
            Span::styled("  备注:       ", Style::default().fg(tc.muted)),
            Span::styled(
                "基于当前 llmfit 匹配度/速度启发式算法进行估算。",
                Style::default().fg(tc.warning),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  输入项 (可编辑)",
            Style::default().fg(tc.accent),
        )),
        Line::from(vec![
            Span::styled("  上下文:     ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_context_input.is_empty() {
                    "<必填>"
                } else {
                    app.plan_context_input.as_str()
                },
                field_style(PlanField::Context),
            ),
            Span::styled(" tokens", Style::default().fg(tc.muted)),
        ]),
        Line::from(vec![
            Span::styled("  量化:       ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_quant_input.is_empty() {
                    "<自动>"
                } else {
                    app.plan_quant_input.as_str()
                },
                field_style(PlanField::Quant),
            ),
        ]),
        Line::from(vec![
            Span::styled("  目标 TPS:   ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_target_tps_input.is_empty() {
                    "<无>"
                } else {
                    app.plan_target_tps_input.as_str()
                },
                field_style(PlanField::TargetTps),
            ),
            Span::styled(" tok/s", Style::default().fg(tc.muted)),
        ]),
        Line::from(""),
    ];

    if let Some(err) = &app.plan_error {
        lines.push(Line::from(vec![
            Span::styled("  Error: ", Style::default().fg(tc.error)),
            Span::styled(err, Style::default().fg(tc.error).bold()),
        ]));
    } else if let Some(plan) = &app.plan_estimate {
        lines.push(Line::from(Span::styled(
            "  Minimum Hardware",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(vec![
            Span::styled("  显存: ", Style::default().fg(tc.muted)),
            Span::styled(
                plan.minimum
                    .vram_gb
                    .map(|v| format!("{v:.1} GB"))
                    .unwrap_or_else(|| "无".to_string()),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   内存: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", plan.minimum.ram_gb),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   CPU: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} 核", plan.minimum.cpu_cores),
                Style::default().fg(tc.fg),
            ),
        ]));
        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  推荐硬件配置",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(vec![
            Span::styled("  显存: ", Style::default().fg(tc.muted)),
            Span::styled(
                plan.recommended
                    .vram_gb
                    .map(|v| format!("{v:.1} GB"))
                    .unwrap_or_else(|| "无".to_string()),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   内存: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", plan.recommended.ram_gb),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   CPU: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} 核", plan.recommended.cpu_cores),
                Style::default().fg(tc.fg),
            ),
        ]));
        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  运行路径",
            Style::default().fg(tc.accent),
        )));

        for path in &plan.run_paths {
            let path_color = if path.feasible { tc.good } else { tc.error };
            let status = if path.feasible { "是" } else { "否" };
            lines.push(Line::from(vec![
                Span::styled("  - ", Style::default().fg(tc.muted)),
                Span::styled(path.path.label(), Style::default().fg(tc.fg).bold()),
                Span::styled(": ", Style::default().fg(tc.muted)),
                Span::styled(status, Style::default().fg(path_color)),
                Span::styled("  tps=", Style::default().fg(tc.muted)),
                Span::styled(
                    path.estimated_tps
                        .map(|t| format!("{t:.1}"))
                        .unwrap_or_else(|| "-".to_string()),
                    Style::default().fg(tc.fg),
                ),
                Span::styled("  匹配=", Style::default().fg(tc.muted)),
                Span::styled(
                    path.fit_level
                        .map(|f| match f {
                            FitLevel::Perfect => "完美",
                            FitLevel::Good => "良好",
                            FitLevel::Marginal => "勉强",
                            FitLevel::TooTight => "内存不足",
                        })
                        .unwrap_or("-"),
                    Style::default().fg(path_color),
                ),
            ]));
        }

        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  升级建议",
            Style::default().fg(tc.accent),
        )));
        if plan.upgrade_deltas.is_empty() {
            lines.push(Line::from(Span::styled(
                "  - 无需升级硬件",
                Style::default().fg(tc.good),
            )));
        } else {
            for delta in &plan.upgrade_deltas {
                lines.push(Line::from(Span::styled(
                    format!("  - {}", delta.description),
                    Style::default().fg(tc.fg),
                )));
            }
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(format!(" 规划: {} ", model_name))
        .title_style(Style::default().fg(tc.fg).bold());

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_provider_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.providers.iter().map(|p| p.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.providers.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.providers.len();

    let scroll_offset = if app.provider_cursor >= inner_height {
        app.provider_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .providers
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_providers[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.provider_cursor;

            let style = if is_cursor {
                if app.selected_providers[i] {
                    Style::default()
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_providers[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_providers.iter().filter(|&&s| s).count();
    let title = format!(" 提供商 ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_use_case_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app
        .use_cases
        .iter()
        .map(|uc| uc.label().len())
        .max()
        .unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.use_cases.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.use_cases.len();

    let scroll_offset = if app.use_case_cursor >= inner_height {
        app.use_case_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .use_cases
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, use_case)| {
            let checkbox = if app.selected_use_cases[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.use_case_cursor;

            let style = if is_cursor {
                if app.selected_use_cases[i] {
                    Style::default()
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_use_cases[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(
                format!(" {} {}", checkbox, use_case.label()),
                style,
            ))
        })
        .collect();

    let active_count = app.selected_use_cases.iter().filter(|&&s| s).count();
    let title = format!(" 用途 ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_capability_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app
        .capabilities
        .iter()
        .map(|c| c.label().len())
        .max()
        .unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.capabilities.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.capabilities.len();

    let scroll_offset = if app.capability_cursor >= inner_height {
        app.capability_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .capabilities
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, cap)| {
            let checkbox = if app.selected_capabilities[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.capability_cursor;

            let style = if is_cursor {
                if app.selected_capabilities[i] {
                    Style::default()
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_capabilities[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(
                format!(" {} {}", checkbox, cap.label()),
                style,
            ))
        })
        .collect();

    let active_count = app.selected_capabilities.iter().filter(|&&s| s).count();
    let title = format!(" 能力 ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_download_provider_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();
    let popup_width = 44.min(area.width.saturating_sub(4));
    let popup_height = 8.min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let mut lines = Vec::new();
    if let Some(name) = &app.download_provider_model {
        lines.push(Line::from(Span::styled(
            format!(" 模型: {}", name),
            Style::default().fg(tc.muted),
        )));
        lines.push(Line::from(""));
    }

    for (i, provider) in app.download_provider_options.iter().enumerate() {
        let label = match provider {
            DownloadProvider::Ollama => "Ollama",
            DownloadProvider::LlamaCpp => "llama.cpp",
        };
        let is_cursor = i == app.download_provider_cursor;
        let prefix = if is_cursor { ">" } else { " " };
        let style = if is_cursor {
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD)
                .bg(tc.highlight_bg)
        } else {
            Style::default().fg(tc.fg)
        };
        lines.push(Line::from(Span::styled(
            format!(" {} {}", prefix, label),
            style,
        )));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(" 下载方式 ")
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn status_keys_and_mode(app: &App) -> (String, String) {
    match app.input_mode {
        InputMode::Normal => {
            if app.show_multi_compare {
                return (" ←/→/hl:滚动  q/Esc:关闭".to_string(), "对比".to_string());
            }
            let detail_key = if app.show_detail {
                "Enter:表格"
            } else {
                "Enter:详情"
            };
            let any_provider = app.ollama_available || app.mlx_available || app.llamacpp_available;
            let ollama_keys = if any_provider {
                let installed_key = if app.installed_first {
                    "i:全部"
                } else {
                    "i:已安装↑"
                };
                format!("  {}  d:下载  r:刷新", installed_key)
            } else {
                String::new()
            };
            (
                format!(
                    " ↑↓/jk:导航  {}  /:搜索  f:匹配屏  s:排序  v:选中模式  V:单列过滤  t:主题  p:规划  m:标记  c:对比  x:清除标记{}  P:提供商  U:用途  C:能力  q:退出  tok/s*:预估",
                    detail_key, ollama_keys,
                ),
                "常规".to_string(),
            )
        }
        InputMode::Visual => {
            let count = app.visual_selection_count();
            (
                format!(
                    " ↑↓/jk:扩展选区  c:对比选区  m:标记  Esc:退出  (已选 {})",
                    count
                ),
                "多选".to_string(),
            )
        }
        InputMode::Select => {
            let header_names = [
                "",
                "安装",
                "模型",
                "提供商",
                "参数",
                "评分",
                "tok/s*",
                "量化",
                "模式",
                "内存 %",
                "上下文",
                "日期",
                "匹配",
                "用途",
            ];
            let col_name = header_names.get(app.select_column).unwrap_or(&"");
            (
                format!(
                    " ←/→:选择列  ↑↓:导航  Enter:过滤列 [{}]  Esc:退出",
                    col_name
                ),
                "单列过滤".to_string(),
            )
        }
        InputMode::Search => (
            "  输入搜索词  Esc:完成  Ctrl-U:清空".to_string(),
            "搜索".to_string(),
        ),
        InputMode::Plan => (
            "  Tab/jk:切换字段  ←/→:光标  字符:编辑  Backspace/Delete:删除  Ctrl-U:清空  Esc:退出"
                .to_string(),
            "规划".to_string(),
        ),
        InputMode::ProviderPopup => (
            "  ↑↓/jk:导航  Space:勾选  a:全选/反选  Esc:离开".to_string(),
            "提供商过滤".to_string(),
        ),
        InputMode::UseCasePopup => (
            "  ↑↓/jk:导航  Space:勾选  a:全选/反选  Esc:离开".to_string(),
            "用途过滤".to_string(),
        ),
        InputMode::CapabilityPopup => (
            "  ↑↓/jk:导航  Space:勾选  a:全选/反选  Esc:离开".to_string(),
            "能力过滤".to_string(),
        ),
        InputMode::DownloadProviderPopup => (
            "  ↑↓/jk:选择  Enter:开始下载  Esc:取消".to_string(),
            "选择下载".to_string(),
        ),
        InputMode::QuantPopup => (
            "  ↑↓/jk:导航  Space:勾选  a:全选/反选  Esc:离开".to_string(),
            "量化过滤".to_string(),
        ),
        InputMode::RunModePopup => (
            "  ↑↓/jk:导航  Space:勾选  a:全选/反选  Esc:离开".to_string(),
            "模式过滤".to_string(),
        ),
        InputMode::ParamsBucketPopup => (
            "  ↑↓/jk:导航  Space:勾选  a:全选/反选  Esc:离开".to_string(),
            "参数过滤".to_string(),
        ),
    }
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let (keys, mode_text) = status_keys_and_mode(app);

    // If a download is in progress, show the progress bar
    if let Some(status) = &app.pull_status {
        let progress_text = if let Some(pct) = app.pull_percent {
            format!(" {} [{:.0}%] ", status, pct)
        } else {
            format!(" {} ", status)
        };

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(progress_text.len() as u16 + 2),
            ])
            .split(area);

        let status_line = Line::from(vec![
            Span::styled(
                format!(" {} ", mode_text),
                Style::default().fg(tc.status_fg).bg(tc.status_bg).bold(),
            ),
            Span::styled(keys, Style::default().fg(tc.muted)),
        ]);
        frame.render_widget(Paragraph::new(status_line), chunks[0]);

        let pull_color = if app.pull_active.is_some() {
            tc.warning
        } else {
            tc.good
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                progress_text,
                Style::default().fg(pull_color),
            ))),
            chunks[1],
        );
        return;
    }

    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", mode_text),
            Style::default().fg(tc.status_fg).bg(tc.status_bg).bold(),
        ),
        Span::styled(keys, Style::default().fg(tc.muted)),
    ]);

    frame.render_widget(Paragraph::new(status_line), area);
}

fn draw_quant_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.quants.iter().map(|q| q.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.quants.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.quants.len();

    let scroll_offset = if app.quant_cursor >= inner_height {
        app.quant_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .quants
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_quants[i] { "[x]" } else { "[ ]" };
            let is_cursor = i == app.quant_cursor;

            let style = if is_cursor {
                if app.selected_quants[i] {
                    Style::default()
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_quants[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_quants.iter().filter(|&&s| s).count();
    let title = format!(" 量化等级 ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_run_mode_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.run_modes.iter().map(|m| m.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.run_modes.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.run_modes.len();

    let scroll_offset = if app.run_mode_cursor >= inner_height {
        app.run_mode_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .run_modes
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_run_modes[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.run_mode_cursor;

            let style = if is_cursor {
                if app.selected_run_modes[i] {
                    Style::default()
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_run_modes[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_run_modes.iter().filter(|&&s| s).count();
    let title = format!(" 运行模式 ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_params_bucket_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app
        .params_buckets
        .iter()
        .map(|b| b.len())
        .max()
        .unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.params_buckets.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.params_buckets.len();

    let scroll_offset = if app.params_bucket_cursor >= inner_height {
        app.params_bucket_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .params_buckets
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_params_buckets[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.params_bucket_cursor;

            let style = if is_cursor {
                if app.selected_params_buckets[i] {
                    Style::default()
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_params_buckets[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_params_buckets.iter().filter(|&&s| s).count();
    let title = format!(" 参数规模 ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}
