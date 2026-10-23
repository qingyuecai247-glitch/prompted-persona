# ============================================================================
# RQ1 reliability heatmaps
#





# ============================================================================

project_lib <- Sys.getenv("PERSONA_R_LIBRARY", unset = "")
if (nzchar(project_lib)) {
  if (!dir.exists(project_lib)) stop("Configured R library does not exist: ", project_lib)
  .libPaths(c(project_lib, .libPaths()))
}

cat("当前 R 包路径：\n")
print(.libPaths())

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(RColorBrewer)


if (!requireNamespace("lme4", quietly = TRUE)) {
  stop(
    "缺少 lme4 包。请先运行：\n",
    "install.packages('lme4', repos = 'https://cloud.r-project.org')"
  )
}

# ============================================================================

# ============================================================================

base_dir <- Sys.getenv("PERSONA_DATA_ROOT", unset = file.path(getwd(), "data", "raw"))

reference_dir <- Sys.getenv("PERSONA_REFERENCE_ROOT", unset = file.path(getwd(), "data", "reference"))
summary_path <- file.path(reference_dir, "analysis", "results", "reliability_all_models.csv")
data_dir <- file.path(base_dir, "llm_output")
output_dir <- Sys.getenv("PERSONA_FIGURE_OUTPUT", unset = file.path(getwd(), "outputs", "figures"))

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}

if (!file.exists(summary_path)) {
  stop(
    "找不到可靠性汇总文件：\n",
    summary_path
  )
}

# ============================================================================

# ============================================================================

TRAITS <- c(
  "State_E",
  "State_A",
  "State_C",
  "State_N",
  "State_O"
)

trait_labels <- c(
  "State_E" = "E",
  "State_A" = "A",
  "State_C" = "C",
  "State_N" = "N",
  "State_O" = "O"
)

SITUATIONS <- c(
  "pos",
  "neg",
  "soc",
  "dut",
  "int",
  "adv",
  "dec",
  "mat"
)

MODEL_ORDER <- c(
  "Human",
  "GPT-4o-mini",
  "GPT-5.2",
  "Qwen-2.5-7B",
  "Qwen-2.5-14B",
  "Qwen-2.5-32B",
  "Grok-3",
  "DeepSeek-3.2",
  "Llama-3.1-8B",
  "Phi-mini",
  "Phi-small",
  "Phi-3-medium"
)



MODEL_FILES <- c(
  "Qwen-2.5-14B" =
    "simulation_output_flattened-qwen2.5-14b-instruct.csv",

  "Phi-mini" =
    "simulation_output_flattened-phi3-mini.csv",

  "Phi-small" =
    "simulation_output_flattened-phi3-small.csv",

  "Phi-medium" =
    "simulation_output_flattened-phi3-medium.csv"
)

# ============================================================================

# ============================================================================

standardize_group <- function(x) {

  y <- trimws(
    as.character(x)
  )
  y <- gsub("[−–—]", "-", y)

  dplyr::case_when(
    grepl(
      "^human$",
      y,
      ignore.case = TRUE
    ) ~ "Human",

    grepl(
      "gpt[- ]?4o",
      y,
      ignore.case = TRUE
    ) ~ "GPT-4o-mini",

    grepl(
      "gpt[- ]?5\\.?2|gpt52",
      y,
      ignore.case = TRUE
    ) ~ "GPT-5.2",

    grepl(
      "qwen.*0\\.5",
      y,
      ignore.case = TRUE
    ) ~ "Qwen-2.5-0.5B",

    grepl(
      "qwen.*1\\.5",
      y,
      ignore.case = TRUE
    ) ~ "Qwen-2.5-1.5B",

    grepl(
      "qwen.*3b|qwen-3b",
      y,
      ignore.case = TRUE
    ) ~ "Qwen-2.5-3B",

    grepl(
      "qwen.*7",
      y,
      ignore.case = TRUE
    ) ~ "Qwen-2.5-7B",

    grepl(
      "qwen.*14",
      y,
      ignore.case = TRUE
    ) ~ "Qwen-2.5-14B",

    grepl(
      "qwen.*32",
      y,
      ignore.case = TRUE
    ) ~ "Qwen-2.5-32B",

    grepl(
      "grok",
      y,
      ignore.case = TRUE
    ) ~ "Grok-3",

    grepl(
      "deepseek",
      y,
      ignore.case = TRUE
    ) ~ "DeepSeek-3.2",

    grepl(
      "llama",
      y,
      ignore.case = TRUE
    ) ~ "Llama-3.1-8B",

    grepl(
      "phi.*mini",
      y,
      ignore.case = TRUE
    ) ~ "Phi-mini",

    grepl(
      "phi.*small",
      y,
      ignore.case = TRUE
    ) ~ "Phi-small",

    grepl(
      "phi.*medium",
      y,
      ignore.case = TRUE
    ) ~ "Phi-3-medium",

    TRUE ~ y
  )
}

# ============================================================================

# ============================================================================

calculate_reliability <- function(
  data,
  trait_var,
  situation_var
) {

  temp_data <- data %>%
    select(
      id_p,
      all_of(trait_var),
      all_of(situation_var)
    ) %>%
    filter(
      is.finite(.data[[trait_var]]),
      is.finite(.data[[situation_var]])
    ) %>%
    group_by(id_p) %>%
    filter(
      n() >= 2
    ) %>%
    ungroup()

  
  if (nrow(temp_data) < 100) {
    warning(
      "数据量不足：",
      trait_var,
      " × ",
      situation_var,
      "，有效行数为 ",
      nrow(temp_data)
    )
    return(NA_real_)
  }

  
  sigma_x2 <- var(
    temp_data[[situation_var]],
    na.rm = TRUE
  )

  if (
    !is.finite(sigma_x2) ||
    sigma_x2 <= 0
  ) {
    warning(
      "情境变量无有效方差：",
      situation_var
    )
    return(NA_real_)
  }

  formula_now <- as.formula(
    paste0(
      trait_var,
      " ~ ",
      situation_var,
      " + (1 + ",
      situation_var,
      " | id_p)"
    )
  )

  model <- tryCatch(
    {
      lme4::lmer(
        formula_now,
        data = temp_data,
        REML = TRUE,
        control = lme4::lmerControl(
          optimizer = "bobyqa",
          optCtrl = list(
            maxfun = 100000
          )
        )
      )
    },
    error = function(e) {
      warning(
        "模型拟合失败：",
        trait_var,
        " × ",
        situation_var,
        "；",
        conditionMessage(e)
      )
      NULL
    }
  )

  if (is.null(model)) {
    return(NA_real_)
  }

  vc <- lme4::VarCorr(model)

  if (!"id_p" %in% names(vc)) {
    warning(
      "模型中没有 id_p 随机效应：",
      trait_var,
      " × ",
      situation_var
    )
    return(NA_real_)
  }

  random_effect_matrix <- as.matrix(
    vc$id_p
  )

  if (
    !situation_var %in% rownames(random_effect_matrix) ||
    !situation_var %in% colnames(random_effect_matrix)
  ) {
    warning(
      "没有找到随机斜率方差：",
      trait_var,
      " × ",
      situation_var
    )
    return(NA_real_)
  }

  sigma2 <- random_effect_matrix[
    situation_var,
    situation_var
  ]

  sigma_g2 <- sigma(model)^2
  avg_t1 <- mean(
    table(temp_data$id_p)
  )

  denominator_component <-
    (avg_t1 - 1) *
    sigma_x2

  if (
    !is.finite(sigma2) ||
    !is.finite(sigma_g2) ||
    !is.finite(denominator_component) ||
    denominator_component <= 0
  ) {
    return(NA_real_)
  }

  rel <- sigma2 / (
    sigma2 +
      sigma_g2 /
      denominator_component
  )

  if (!is.finite(rel)) {
    return(NA_real_)
  }

  max(
    0,
    min(
      1,
      rel
    )
  )
}

# ============================================================================

# ============================================================================

calculate_model_file <- function(
  model_name,
  file_name
) {

  path <- file.path(
    data_dir,
    file_name
  )

  if (!file.exists(path)) {
    stop(
      "模型 ",
      model_name,
      " 的真实数据文件不存在：\n",
      path
    )
  }

  cat(
    "\n正在从真实数据重新计算：",
    model_name,
    "\n文件：",
    path,
    "\n"
  )

  raw <- read.csv(
    path,
    stringsAsFactors = FALSE,
    fileEncoding = "UTF-8-BOM",
    check.names = FALSE
  )

  required_columns <- c(
    "id_p",
    TRAITS,
    SITUATIONS
  )

  missing_columns <- setdiff(
    required_columns,
    names(raw)
  )

  if (length(missing_columns) > 0) {
    stop(
      "模型 ",
      model_name,
      " 的数据缺少以下字段：\n",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  }

  raw$id_p <- as.factor(
    raw$id_p
  )

  numeric_columns <- c(
    TRAITS,
    SITUATIONS
  )

  for (col in numeric_columns) {
    raw[[col]] <- suppressWarnings(
      as.numeric(raw[[col]])
    )
  }

  result <- expand.grid(
    trait = TRAITS,
    situation = SITUATIONS,
    stringsAsFactors = FALSE
  ) %>%
    rowwise() %>%
    mutate(
      Group = model_name,
      reliability = calculate_reliability(
        raw,
        trait,
        situation
      )
    ) %>%
    ungroup() %>%
    select(
      Group,
      trait,
      situation,
      reliability
    )

  n_valid <- sum(
    is.finite(result$reliability)
  )

  cat(
    model_name,
    "：成功得到 ",
    n_valid,
    " / ",
    nrow(result),
    " 个有效可靠性值。\n"
  )

  if (n_valid == 0) {
    stop(
      "模型 ",
      model_name,
      " 的真实数据已读取，但 40 个可靠性值全部计算失败。请检查数据结构或 lme4 拟合警告。"
    )
  }

  result
}

# ============================================================================

# ============================================================================

df_summary <- read.csv(
  summary_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required_summary_columns <- c(
  "Group",
  "trait",
  "situation",
  "reliability"
)

missing_summary_columns <- setdiff(
  required_summary_columns,
  names(df_summary)
)

if (length(missing_summary_columns) > 0) {
  stop(
    "汇总文件缺少以下字段：\n",
    paste(
      missing_summary_columns,
      collapse = ", "
    )
  )
}

df_summary <- df_summary %>%
  transmute(
    Group = standardize_group(Group),
    trait = as.character(trait),
    situation = as.character(situation),
    reliability = suppressWarnings(
      as.numeric(reliability)
    )
  )

# ============================================================================

# ============================================================================

models_to_recalculate <- names(
  MODEL_FILES
)


real_results <- bind_rows(
  lapply(
    models_to_recalculate,
    function(model_name) {
      calculate_model_file(
        model_name = model_name,
        file_name = MODEL_FILES[[model_name]]
      )
    }
  )
)



df <- df_summary %>%
  filter(
    !Group %in% models_to_recalculate
  ) %>%
  bind_rows(
    real_results
  ) %>%
  mutate(
    Group = standardize_group(Group)
  ) %>%
  group_by(
    Group,
    trait,
    situation
  ) %>%
  summarise(
    reliability = {
      valid_values <- reliability[
        is.finite(reliability)
      ]

      if (length(valid_values) == 0) {
        NA_real_
      } else {
        valid_values[
          length(valid_values)
        ]
      }
    },
    .groups = "drop"
  )


phi_medium_check <- df %>%
  filter(
    Group == "Phi-3-medium"
  )

phi_medium_valid_n <- sum(
  is.finite(phi_medium_check$reliability)
)

if (nrow(phi_medium_check) == 0) {
  stop(
    "Phi-medium 没有进入最终数据。"
  )
}

if (phi_medium_valid_n == 0) {
  stop(
    "Phi-medium 已读取真实 CSV，但没有任何有效可靠性结果，因此停止作图。"
  )
}

cat(
  "\nPhi-medium 已使用真实数据：",
  phi_medium_valid_n,
  " / 40 个有效可靠性值。\n"
)


updated_summary_path <- file.path(
  output_dir,
  "calculated_reliability_results_with_real_phi_medium.csv"
)

write.csv(
  df,
  updated_summary_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat(
  "更新后的可靠性结果已保存至：\n",
  updated_summary_path,
  "\n"
)

# ============================================================================

# ============================================================================

grid_template <- expand.grid(
  Group = MODEL_ORDER,
  trait = TRAITS,
  situation = SITUATIONS,
  stringsAsFactors = FALSE
)

df_plot <- grid_template %>%
  left_join(
    df,
    by = c(
      "Group",
      "trait",
      "situation"
    )
  ) %>%
  mutate(
    Group = factor(
      Group,
      levels = MODEL_ORDER
    ),
    Trait_Short = factor(
      trait_labels[trait],
      levels = rev(
        c(
          "E",
          "A",
          "C",
          "N",
          "O"
        )
      )
    ),
    situation = factor(
      situation,
      levels = SITUATIONS
    )
  )


coverage_check <- df_plot %>%
  group_by(Group) %>%
  summarise(
    Valid_cells = sum(
      is.finite(reliability)
    ),
    Total_cells = n(),
    .groups = "drop"
  )

print(
  coverage_check
)


phi_plot_coverage <- coverage_check %>%
  filter(
    Group == "Phi-3-medium"
  ) %>%
  pull(Valid_cells)

if (
  length(phi_plot_coverage) == 0 ||
  phi_plot_coverage == 0
) {
  stop(
    "Phi-medium 在作图数据中仍没有有效值，停止输出，避免生成占位图。"
  )
}

# ============================================================================

# ============================================================================

create_heatmap <- function(
  model_name,
  show_x = TRUE
) {

  sub_data <- df_plot %>%
    filter(
      Group == model_name
    )

  valid_cell_count <- sum(
    is.finite(sub_data$reliability)
  )

  
  
  if (valid_cell_count == 0) {
    stop(
      "模型 ",
      model_name,
      " 没有任何有效可靠性结果，已停止作图，避免生成占位图。"
    )
  }

  p <- ggplot(
    sub_data,
    aes(
      x = situation,
      y = Trait_Short,
      fill = reliability
    )
  ) +
    geom_tile(
      color = "white",
      linewidth = 1
    ) +
    scale_fill_gradient(
      low = "white",
      high = "#BE4A45",
      limits = c(0, 1),
      na.value = "#F2F2F2",
      name = "Reliability"
    ) +
    scale_x_discrete(
      position = "top"
    ) +
    labs(
      title = model_name,
      x = NULL,
      y = NULL
    ) +
    theme_minimal() +
    theme(
      text = element_text(
        family = "sans"
      ),
      plot.title = element_text(
        hjust = 0.5,
        face = "bold",
        size = 17,
        color = "black",
        margin = margin(
          b = 4,
          unit = "pt"
        )
      ),
      axis.text.y = element_text(
        size = 15,
        face = "bold"
      ),
      panel.grid = element_blank(),
      aspect.ratio = 5 / 8,
      plot.margin = margin(
        2,
        2,
        2,
        2,
        "pt"
      )
    )

  if (show_x) {
    p <- p +
      theme(
        axis.text.x = element_text(
          angle = 45,
          hjust = 0,
          size = 15,
          face = "bold"
        )
      )
  } else {
    p <- p +
      theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()
      )
  }

  p <- p +
    theme(
      panel.background = element_rect(
        fill = "white",
        color = "#DDDDDD",
        linewidth = 0.5
      ),
      plot.background = element_rect(
        fill = "white",
        color = NA
      )
    )

  p
}

# ============================================================================

# ============================================================================

render_heatmap_layout <- function(
  ncol_layout,
  output_name,
  width,
  height,
  model_order = MODEL_ORDER,
  total_slots = length(model_order)
) {

  plot_list <- lapply(seq_along(model_order), function(i) {
    p <- create_heatmap(
      model_order[i],
      show_x = i <= ncol_layout
    )

    p
  })

  
  
  if (total_slots > length(plot_list)) {
    plot_list <- c(
      plot_list,
      rep(
        list(
          plot_spacer()
        ),
        total_slots - length(plot_list)
      )
    )
  }

  combined_plot <- wrap_plots(
    plot_list,
    ncol = ncol_layout,
    heights = c(
      1.03,
      rep(
        1,
        max(0, ceiling(total_slots / ncol_layout) - 1)
      )
    ),
    guides = "collect"
  ) +
    plot_annotation() &
    theme(
      legend.position = "right",
      legend.title = element_text(
        size = 16,
        face = "bold"
      ),
      legend.text = element_text(
        size = 15
      ),
      legend.box.margin = margin(
        0,
        0,
        0,
        18,
        "pt"
      ),
      plot.margin = margin(
        2,
        18,
        2,
        2,
        "pt"
      )
    )

  if (interactive()) {
    print(
      combined_plot
    )
  }

  output_path <- file.path(
    output_dir,
    output_name
  )

  ggsave(
    output_path,
    combined_plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )

  cat(
    "图片已保存至：\n",
    output_path,
    "\n"
  )
}

# ============================================================================

# ============================================================================

if (
  !exists("RQ1_SKIP_DEFAULT") ||
  !RQ1_SKIP_DEFAULT
) {
  render_heatmap_layout(
    ncol_layout = 4,
    output_name = "RQ_8_real_phi_medium.pdf",
    width = 16,
    height = 9.2,
    total_slots = 12
  )

  selected_model_order <- c(
    "Human",
    "GPT-4o-mini",
    "GPT-5.2",
    "Qwen-2.5-32B",
    "Grok-3",
    "DeepSeek-3.2",
    "Llama-3.1-8B",
    "Phi-3-medium"
  )

  render_heatmap_layout(
    ncol_layout = 4,
    output_name = "RQ_8_selected_models_human.pdf",
    width = 16,
    height = 6.2,
    model_order = selected_model_order,
    total_slots = 8
  )
}
