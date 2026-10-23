# Ordered Phi-medium display variant of Figure 3 after explicitly
# mean-centring each DIAMONDS dimension. The displayed order is Human,
# DeepSeek-3.2, Grok-3, GPT-5.2, Qwen2.5-32B-Instruct,
# Phi-3-medium-4k-instruct, GPT-4o-mini and Llama-3.1-8B-Instruct.
# across the 62 situation profiles used in the analysis.
#
# The centring rule is x_c[j,k] = x[j,k] - mean_j(x[j,k]). Because the
# centred scores sum to zero across situations, mean(x_c * State) is free of
# any additive contribution from a persona's mean state level.
#
# The script writes a centring audit, centred person-level SCSCs, factor
# loadings, FMI estimates/bootstrap intervals, an old-versus-new FMI
# comparison, and publication figure exports without overwriting the legacy
# uncentred analysis.
# The upper panels retain the original loading heatmap + trait-flow design.
# The lower summary uses the same minres/oblimin loadings and |loading| > .50
# definition as the formal factor-diagnostics analysis.

suppressPackageStartupMessages({
  library(psych)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

set.seed(20260721)

# Explicit roots; full refits write separately from archived reference results.
repo_dir <- normalizePath(Sys.getenv("PERSONA_REPO_ROOT", unset = getwd()), mustWork = TRUE)
data_dir <- normalizePath(Sys.getenv("LLM_DATA_DIR", unset = file.path(repo_dir, "data", "raw", "llm_output")), mustWork = TRUE)
project_dir <- Sys.getenv("MANUSCRIPT_PROJECT_DIR", unset = file.path(repo_dir, "outputs", "factor_refit"))
dir.create(project_dir, recursive = TRUE, showWarnings = FALSE)
project_dir <- normalizePath(project_dir, mustWork = TRUE)
output_dir <- file.path(project_dir, "Figures_1", "factor_modularity_centered_diamonds_phi_medium_ordered")
source_dir <- file.path(project_dir, "analysis", "results", "factor_modularity_centered_diamonds_phi_medium")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

file_names <- c(
  human = "dat_aggregated_final.csv",
  gpt4omini = "simulation_output_flattened-gpt4omini.csv",
  gpt52 = "simulation_output_flattened-gpt52.csv",
  qwen7 = "simulation_output_flattened-qwen2.5-7b-instruct.csv",
  qwen14 = "simulation_output_flattened-qwen2.5-14b-instruct.csv",
  qwen32 = "simulation_output_flattened-qwen2.5-32b-instruct.csv",
  grok3 = "simulation_output_flattened-grok3.csv",
  deepseek = "simulation_output_flattened-deepseek3.2.csv",
  llama = "simulation_output_flattened_llama3.1-8b-instruct.csv",
  phimini = "simulation_output_flattened-phi3-mini.csv",
  phismall = "simulation_output_flattened-phi3-small.csv",
  phimed = "simulation_output_flattened-phi3-medium.csv"
)
files <- setNames(file.path(data_dir, unname(file_names)), names(file_names))

model_labels <- c(
  human = "Human",
  gpt4omini = "GPT-4o-mini",
  gpt52 = "GPT-5.2",
  qwen7 = "Qwen2.5-7B-Instruct",
  qwen14 = "Qwen2.5-14B-Instruct",
  qwen32 = "Qwen2.5-32B",
  grok3 = "Grok-3",
  deepseek = "DeepSeek-3.2",
  llama = "Llama-3.1-8B",
  phimini = "Phi-3-mini-4k-instruct",
  phismall = "Phi-3-small-8k-instruct",
  phimed = "Phi-3-medium"
)

representatives <- c(
  "human", "deepseek", "grok3", "gpt52",
  "qwen32", "phimed", "gpt4omini", "llama"
)

trait_levels <- c("O", "C", "E", "A", "N")
states <- c("E", "A", "C", "N", "O")
diamonds <- c("dut", "int", "adv", "mat", "pos", "neg", "dec", "soc")
trait_colours <- c(
  O = "#D1494E", C = "#3478B7", E = "#F08A18",
  A = "#4FA64F", N = "#8255A6"
)
loading_threshold_flow <- 0.30
loading_threshold_fmi <- 0.50
base_family <- "Arial"
show_trait_group_bands <- TRUE

read_valid_rows <- function(file_path) {
  dat <- read.csv(file_path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c("id_p", diamonds, paste0("State_", states))
  missing <- setdiff(required, names(dat))
  if (length(missing)) {
    stop("Missing columns in ", basename(file_path), ": ", paste(missing, collapse = ", "))
  }
  valid_id <- !is.na(dat$id_p) & nzchar(trimws(as.character(dat$id_p)))
  dat[valid_id, , drop = FALSE]
}

extract_situation_profiles <- function(file_path) {
  dat <- read_valid_rows(file_path)
  profiles <- unique(dat[, diamonds, drop = FALSE])
  profiles <- profiles[complete.cases(profiles), , drop = FALSE]
  if (nrow(profiles) != 62L) {
    stop("Expected 62 unique DIAMONDS profiles in ", basename(file_path),
         "; found ", nrow(profiles), ".")
  }
  profiles
}

# DIAMONDS scores are normative situation-level quantities shared by all
# participants/personas. Estimate the centring constants once from the 62
# profiles in the human reference file and verify that every model file has
# the same profile means to numerical precision.
reference_profiles <- extract_situation_profiles(files[["human"]])
diamond_means <- colMeans(reference_profiles)

centering_audit <- bind_rows(lapply(names(files), function(key) {
  profiles <- extract_situation_profiles(files[[key]])
  tibble(
    model_key = key,
    dimension = diamonds,
    n_situations = nrow(profiles),
    raw_mean = unname(colMeans(profiles)),
    reference_mean = unname(diamond_means),
    mean_difference = raw_mean - reference_mean,
    centred_mean = unname(colMeans(sweep(profiles, 2, diamond_means, "-")))
  )
}))
write.csv(centering_audit, file.path(source_dir, "diamonds_centering_audit.csv"), row.names = FALSE)
write.csv(
  tibble(dimension = diamonds, mean_across_62_situations = unname(diamond_means)),
  file.path(source_dir, "diamonds_means_62_situations.csv"), row.names = FALSE
)

nearest_pd_correlation <- function(x, eps = 1e-6) {
  nm <- dimnames(x)
  x <- (x + t(x)) / 2
  ev <- eigen(x, symmetric = TRUE)
  values <- pmax(ev$values, eps)
  y <- ev$vectors %*% diag(values) %*% t(ev$vectors)
  d <- sqrt(diag(y))
  y <- y / outer(d, d)
  diag(y) <- 1
  dimnames(y) <- nm
  y
}

build_persona_matrix <- function(file_path) {
  dat <- read_valid_rows(file_path)
  dat[, diamonds] <- sweep(dat[, diamonds, drop = FALSE], 2, diamond_means, "-")

  long <- dat %>%
    mutate(id_p = as.character(id_p)) %>%
    pivot_longer(
      cols = all_of(paste0("State_", states)),
      names_to = "state", values_to = "state_value"
    ) %>%
    mutate(state = sub("State_", "", state)) %>%
    pivot_longer(
      cols = all_of(diamonds),
      names_to = "diamond", values_to = "diamond_value"
    ) %>%
    mutate(value = state_value * diamond_value,
           item = paste("Sit", state, diamond, sep = "_")) %>%
    group_by(id_p, item) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = item, values_from = value)

  ids <- long$id_p
  x <- as.data.frame(long[, setdiff(names(long), "id_p"), drop = FALSE])
  keep <- vapply(x, function(z) sum(is.finite(z)) >= 3 && sd(z, na.rm = TRUE) > 0, logical(1))
  x <- x[, keep, drop = FALSE]
  rownames(x) <- ids
  x
}

fit_factor_model <- function(persona_matrix) {
  cor_mat <- cor(persona_matrix, use = "pairwise.complete.obs")
  cor_mat <- nearest_pd_correlation(cor_mat)
  suppressWarnings(psych::fa(
    cor_mat, nfactors = 5, rotate = "oblimin", fm = "minres",
    warnings = FALSE
  ))
}

compute_factor_diagnostics <- function(persona_matrix, model_key, parallel_reps = 500L) {
  cor_mat <- cor(persona_matrix, use = "pairwise.complete.obs")
  cor_mat <- nearest_pd_correlation(cor_mat)
  kmo <- suppressWarnings(psych::KMO(cor_mat))
  bartlett <- suppressWarnings(psych::cortest.bartlett(cor_mat, n = nrow(persona_matrix)))

  observed_eigenvalues <- sort(eigen(cor_mat, symmetric = TRUE, only.values = TRUE)$values,
                               decreasing = TRUE)
  set.seed(20260721 + match(model_key, names(files)) * 1009L)
  random_eigenvalues <- replicate(parallel_reps, {
    z <- matrix(rnorm(nrow(persona_matrix) * ncol(persona_matrix)),
                nrow = nrow(persona_matrix), ncol = ncol(persona_matrix))
    sort(eigen(cor(z), symmetric = TRUE, only.values = TRUE)$values,
         decreasing = TRUE)
  })
  random_95 <- apply(random_eigenvalues, 1, quantile, probs = .95)
  parallel_n <- sum(observed_eigenvalues > random_95)
  write.csv(
    tibble(
      model_key = model_key,
      component = seq_along(observed_eigenvalues),
      observed_eigenvalue = observed_eigenvalues,
      random_mean_eigenvalue = rowMeans(random_eigenvalues),
      random_95pct_eigenvalue = random_95
    ),
    file.path(source_dir, paste0("parallel_analysis_centered_", model_key, ".csv")),
    row.names = FALSE
  )

  vss <- suppressMessages(suppressWarnings(psych::VSS(
    persona_matrix, n = 8, rotate = "oblimin", fm = "minres", plot = FALSE
  )))
  vss1 <- if (!is.null(vss$cfit.1) && any(is.finite(vss$cfit.1))) which.max(vss$cfit.1) else NA_integer_
  vss2 <- if (!is.null(vss$cfit.2) && any(is.finite(vss$cfit.2))) which.max(vss$cfit.2) else NA_integer_

  tibble(
    model_key = model_key,
    model = unname(model_labels[model_key]),
    n_people_or_personas = nrow(persona_matrix),
    n_scsc_indices = ncol(persona_matrix),
    KMO_overall_MSA = unname(kmo$MSA),
    Bartlett_chisq = unname(bartlett$chisq),
    Bartlett_df = unname(bartlett$df),
    Bartlett_p = unname(bartlett$p.value),
    parallel_iterations = parallel_reps,
    parallel_suggested_n = parallel_n,
    VSS1_suggested_n = vss1,
    VSS2_suggested_n = vss2
  )
}

tidy_loadings <- function(fit, model_key) {
  wide <- as.data.frame(unclass(fit$loadings))
  wide$item <- rownames(wide)
  factor_cols <- setdiff(names(wide), "item")
  factor_map <- setNames(paste0("F", seq_along(factor_cols)), factor_cols)

  wide %>%
    pivot_longer(-item, names_to = "factor_raw", values_to = "loading") %>%
    mutate(
      model_key = model_key,
      model = unname(model_labels[model_key]),
      trait = sub("^Sit_([A-Z])_.*$", "\\1", item),
      factor = factor_map[factor_raw],
      factor_id = match(factor_raw, factor_cols),
      item_id = match(item, unique(item)),
      abs_loading = abs(loading)
    )
}

factor_modularity <- function(loadings_long, threshold = loading_threshold_fmi) {
  per_factor <- loadings_long %>%
    filter(abs_loading > threshold) %>%
    count(factor, trait, name = "n_salient") %>%
    group_by(factor) %>%
    summarise(
      factor_modularity = max(n_salient) / sum(n_salient),
      salient_loadings = sum(n_salient),
      .groups = "drop"
    )
  mean(per_factor$factor_modularity)
}

make_model_panel <- function(loadings_long, model_label) {
  factor_ids <- sort(unique(loadings_long$factor_id))
  item_order <- loadings_long %>%
    distinct(item, trait) %>%
    arrange(factor(trait, levels = trait_levels), item) %>%
    mutate(item_id = row_number())
  pdat <- loadings_long %>%
    select(-item_id) %>%
    left_join(item_order, by = c("item", "trait"))

  trait_y <- item_order %>%
    group_by(trait) %>%
    summarise(y = mean(item_id), .groups = "drop")
  trait_bands <- item_order %>%
    group_by(trait) %>%
    summarise(ymin = min(item_id) - .5, ymax = max(item_id) + .5, .groups = "drop") %>%
    mutate(colour = unname(trait_colours[as.character(trait)]))

  heat <- ggplot()
  if (show_trait_group_bands) {
    heat <- heat + geom_rect(
      data = trait_bands,
      aes(xmin = .05, xmax = max(factor_ids) + .55, ymin = ymin, ymax = ymax),
      fill = trait_bands$colour, alpha = .08, colour = NA, inherit.aes = FALSE
    )
  }
  heat <- heat +
    geom_tile(data = pdat, aes(factor_id, item_id, fill = loading),
              width = .91, height = .91, colour = "white", linewidth = .08)
  if (show_trait_group_bands) {
    heat <- heat + geom_rect(
      data = trait_bands,
      aes(xmin = .05, xmax = max(factor_ids) + .55, ymin = ymin, ymax = ymax),
      fill = trait_bands$colour, alpha = .04, colour = NA, inherit.aes = FALSE
    )
  }
  heat <- heat +
    geom_text(
      data = trait_y, aes(x = .32, y = y, label = trait, colour = trait),
      inherit.aes = FALSE, family = base_family, fontface = "bold", size = 2.20
    ) +
    scale_fill_gradient2(low = "#2F6FB0", mid = "#F8F8F8", high = "#C94545",
                         midpoint = 0, limits = c(-1, 1), guide = "none") +
    scale_colour_manual(values = trait_colours, guide = "none") +
    scale_x_continuous(breaks = factor_ids, labels = paste0("F", factor_ids),
                       limits = c(.05, max(factor_ids) + .55),
                       position = "top", expand = c(0, 0)) +
    scale_y_continuous(limits = c(.5, max(item_order$item_id) + .5), expand = c(0, 0)) +
    theme_void(base_family = base_family) +
    theme(
      axis.text.x = element_text(size = 6.0, face = "bold", colour = "#303030", margin = margin(b = 1)),
      plot.margin = margin(1, 0, 1, 1)
    )

  n_items <- nrow(item_order)
  factor_nodes <- tibble(
    factor_id = factor_ids,
    xend = 1.55,
    yend = seq(n_items * .13, n_items * .87, length.out = length(factor_ids)),
    factor = paste0("F", factor_ids)
  )
  edges <- pdat %>%
    filter(abs_loading > loading_threshold_flow) %>%
    left_join(factor_nodes, by = "factor_id") %>%
    mutate(linewidth = .09 + abs_loading * .42,
           alpha = .28 + abs_loading * .48)

  flow <- ggplot()
  if (show_trait_group_bands) {
    flow <- flow + geom_rect(
      data = trait_bands,
      aes(xmin = -.02, xmax = 1.95, ymin = ymin, ymax = ymax),
      fill = trait_bands$colour, alpha = .07, colour = NA, inherit.aes = FALSE
    )
  }
  flow <- flow +
    geom_curve(
      data = edges,
      aes(x = 0, y = item_id, xend = xend - .04, yend = yend,
          colour = trait, linewidth = linewidth, alpha = alpha),
      curvature = -.30, lineend = "round"
    ) +
    geom_segment(data = factor_nodes,
                 aes(x = xend, xend = xend, y = yend - 1.5, yend = yend + 1.5),
                 linewidth = .18, colour = "#555555") +
    geom_text(data = factor_nodes, aes(x = xend + .05, y = yend, label = factor),
              hjust = 0, family = base_family, fontface = "bold", size = 2.15, colour = "#303030") +
    scale_colour_manual(values = trait_colours, guide = "none") +
    scale_linewidth_identity() + scale_alpha_identity() +
    coord_cartesian(xlim = c(-.02, 1.95), ylim = c(.5, n_items + .5), clip = "off") +
    theme_void(base_family = base_family) +
    theme(plot.margin = margin(1, 2, 1, -1))

  title <- ggplot() +
    annotate("text", x = .5, y = .5, label = model_label, family = base_family,
             fontface = "bold", size = 2.50, colour = "#202020") +
    theme_void() + theme(plot.margin = margin(0, 0, 0, 0))

  title / (heat + flow + plot_layout(widths = c(1.05, .90))) +
    plot_layout(heights = c(.075, 1))
}

bootstrap_fmi <- function(persona_matrix, model_key, reps = 200) {
  cache <- file.path(source_dir, paste0("bootstrap_fmi_", model_key, ".csv"))
  if (file.exists(cache)) return(read.csv(cache, stringsAsFactors = FALSE))
  set.seed(20260721 + match(model_key, names(files)) * 1000003L)
  n <- nrow(persona_matrix)
  vals <- rep(NA_real_, reps)
  for (b in seq_len(reps)) {
    idx <- sample.int(n, n, replace = TRUE)
    fit <- tryCatch(fit_factor_model(persona_matrix[idx, , drop = FALSE]), error = function(e) NULL)
    if (!is.null(fit)) vals[b] <- factor_modularity(tidy_loadings(fit, model_key))
    if (b %% 25 == 0) message(model_key, " bootstrap ", b, "/", reps)
  }
  out <- data.frame(model_key = model_key, replicate = seq_len(reps), FMI = vals)
  write.csv(out, cache, row.names = FALSE)
  out
}

message("Fitting formal factor models...")
persona_data <- lapply(files, build_persona_matrix)
invisible(lapply(names(persona_data), function(key) {
  out <- data.frame(id_p = rownames(persona_data[[key]]), persona_data[[key]], check.names = FALSE)
  write.csv(
    out,
    file.path(source_dir, paste0("person_level_scsc_centered_", key, ".csv")),
    row.names = FALSE
  )
}))
message("Recomputing factorability and factor-number diagnostics...")
factor_diagnostics <- bind_rows(mapply(
  function(x, key) compute_factor_diagnostics(x, key, parallel_reps = 500L),
  persona_data, names(persona_data), SIMPLIFY = FALSE
))
write.csv(
  factor_diagnostics,
  file.path(source_dir, "factor_diagnostics_centered_summary.csv"),
  row.names = FALSE
)
# VSS and parallel analysis consume random numbers. Reset the prespecified seed
# here so that the formal five-factor solutions are reproducible and directly
# comparable with the legacy script regardless of which diagnostics run first.
set.seed(20260721)
fits <- mapply(function(x, key) fit_factor_model(x), persona_data, names(persona_data), SIMPLIFY = FALSE)
names(fits) <- names(persona_data)
loadings <- mapply(function(fit, key) tidy_loadings(fit, key), fits, names(fits), SIMPLIFY = FALSE)
names(loadings) <- names(fits)

fmi_point <- tibble(
  model_key = names(loadings),
  model = unname(model_labels[names(loadings)]),
  n = vapply(persona_data, nrow, integer(1)),
  FMI = vapply(loadings, factor_modularity, numeric(1))
)

message("Bootstrapping FMI uncertainty (cached after the first run)...")
bootstrap <- bind_rows(mapply(
  function(x, key) bootstrap_fmi(x, key, reps = 200),
  persona_data, names(persona_data), SIMPLIFY = FALSE
))
fmi_ci <- bootstrap %>%
  group_by(model_key) %>%
  summarise(
    boot_valid = sum(is.finite(FMI)),
    ci_low = quantile(FMI, .025, na.rm = TRUE),
    ci_high = quantile(FMI, .975, na.rm = TRUE),
    .groups = "drop"
  )
fmi_summary <- fmi_point %>% left_join(fmi_ci, by = "model_key")

write.csv(bind_rows(loadings[representatives]),
          file.path(source_dir, "representative_factor_loadings.csv"), row.names = FALSE)
write.csv(fmi_summary, file.path(source_dir, "factor_modularity_summary.csv"), row.names = FALSE)
write.csv(bootstrap, file.path(source_dir, "factor_modularity_bootstrap.csv"), row.names = FALSE)

# Align centred factors to the legacy solution by maximizing absolute Tucker
# congruence over all 5! factor permutations. This makes factor-number labels
# comparable even if the rotation returns an equivalent permuted solution.
all_permutations <- function(x) {
  if (length(x) == 1L) return(matrix(x, nrow = 1L))
  do.call(rbind, lapply(seq_along(x), function(i) {
    cbind(x[i], all_permutations(x[-i]))
  }))
}

legacy_loadings_path <- file.path(
  project_dir, "analysis", "results", "factor_modularity_corrected",
  "representative_factor_loadings.csv"
)
if (file.exists(legacy_loadings_path)) {
  legacy_loadings <- read.csv(legacy_loadings_path, stringsAsFactors = FALSE)
  stability_rows <- list()
  congruence_rows <- list()
  permutations <- all_permutations(seq_len(5))

  for (key in representatives) {
    if (!key %in% unique(legacy_loadings$model_key)) next
    old <- legacy_loadings %>%
      filter(model_key == key) %>%
      select(item, factor, loading) %>%
      pivot_wider(names_from = factor, values_from = loading)
    new <- loadings[[key]] %>%
      select(item, factor, loading) %>%
      pivot_wider(names_from = factor, values_from = loading)
    common_items <- intersect(old$item, new$item)
    old_mat <- as.matrix(old[match(common_items, old$item), paste0("F", 1:5), drop = FALSE])
    new_mat <- as.matrix(new[match(common_items, new$item), paste0("F", 1:5), drop = FALSE])
    congruence <- psych::factor.congruence(old_mat, new_mat, digits = 10)
    scores <- apply(permutations, 1, function(p) mean(abs(congruence[cbind(1:5, p)])))
    best <- permutations[which.max(scores), ]
    aligned_new <- new_mat[, best, drop = FALSE]
    signs <- sign(diag(psych::factor.congruence(old_mat, aligned_new, digits = 10)))
    signs[signs == 0] <- 1
    aligned_new <- sweep(aligned_new, 2, signs, "*")
    matched_congruence <- diag(psych::factor.congruence(old_mat, aligned_new, digits = 10))

    old_primary <- max.col(abs(old_mat), ties.method = "first")
    new_primary <- max.col(abs(aligned_new), ties.method = "first")
    stability_rows[[key]] <- tibble(
      model_key = key,
      model = unname(model_labels[key]),
      item = common_items,
      old_primary_factor = paste0("F", old_primary),
      centered_primary_factor_aligned = paste0("F", new_primary),
      same_primary_factor = old_primary == new_primary,
      max_abs_loading_change = apply(abs(aligned_new - old_mat), 1, max)
    )
    congruence_rows[[key]] <- tibble(
      model_key = key,
      model = unname(model_labels[key]),
      legacy_factor = paste0("F", 1:5),
      centered_factor_before_alignment = paste0("F", best),
      sign_flipped = signs < 0,
      Tucker_congruence = matched_congruence
    )
  }
  write.csv(
    bind_rows(stability_rows),
    file.path(source_dir, "old_vs_centered_loading_stability.csv"),
    row.names = FALSE
  )
  write.csv(
    bind_rows(congruence_rows),
    file.path(source_dir, "old_vs_centered_tucker_congruence.csv"),
    row.names = FALSE
  )
}

legacy_summary_path <- file.path(
  project_dir, "analysis", "results", "factor_modularity_corrected",
  "factor_modularity_summary.csv"
)
if (file.exists(legacy_summary_path)) {
  legacy_summary <- read.csv(legacy_summary_path, stringsAsFactors = FALSE) %>%
    select(model_key, FMI_old = FMI, ci_low_old = ci_low, ci_high_old = ci_high)
  fmi_comparison <- fmi_summary %>%
    select(model_key, model, FMI_centered = FMI,
           ci_low_centered = ci_low, ci_high_centered = ci_high) %>%
    left_join(legacy_summary, by = "model_key") %>%
    mutate(delta_FMI = FMI_centered - FMI_old)
  write.csv(
    fmi_comparison,
    file.path(source_dir, "old_vs_centered_fmi_comparison.csv"),
    row.names = FALSE
  )
}

panels <- lapply(representatives, function(key) make_model_panel(loadings[[key]], model_labels[[key]]))
top_grid <- wrap_plots(panels, ncol = 4, byrow = TRUE) +
  plot_layout(guides = "collect")

legend_df <- tibble(
  trait = factor(c("N", "A", "E", "C", "O"), levels = trait_levels),
  label = c("N  Neuroticism", "A  Agreeableness", "E  Extraversion",
            "C  Conscientiousness", "O  Openness"),
  x = c(.13, .32, .54, .70, .92),
  y = .60
)
trait_legend <- ggplot(legend_df) +
  annotate("text", x = .01, y = .60, label = "Big Five:", hjust = 0,
           family = base_family, fontface = "bold", size = 2.20, colour = "#303030") +
  geom_point(aes(x, y, colour = trait), size = 1.80) +
  geom_text(aes(x + .014, y, label = label), hjust = 0, family = base_family,
            size = 1.88, colour = "#303030") +
  scale_colour_manual(values = trait_colours, guide = "none") +
  coord_cartesian(xlim = c(0, 1.04), ylim = c(0, 1), clip = "off") +
  theme_void() + theme(plot.margin = margin(0, 1, 0, 1))

loading_scale <- ggplot(tibble(x = seq(-1, 1, length.out = 200), y = .60), aes(x, y, fill = x)) +
  geom_tile(height = .22) +
  scale_fill_gradient2(low = "#2F6FB0", mid = "#F8F8F8", high = "#C94545",
                       midpoint = 0, limits = c(-1, 1), guide = "none") +
  annotate("text", x = c(-1, 0, 1), y = .31, label = c("-1", "0", "1"),
           family = base_family, size = 1.80, colour = "#555555") +
  annotate("text", x = 0, y = .08, label = "Factor loading",
           family = base_family, size = 1.98, colour = "#404040") +
  coord_cartesian(xlim = c(-1, 1), ylim = c(0, 1), clip = "off") +
  theme_void(base_family = base_family) +
  theme(plot.margin = margin(0, 3, 0, 1))

legend_strip <- wrap_plots(
  list(trait_legend, loading_scale), ncol = 2, widths = c(.78, .22)
)

primary_models <- c(
  "deepseek", "grok3", "gpt52", "qwen32",
  "phimed", "gpt4omini", "llama"
)
supplementary_models <- unique(c(
  primary_models, "qwen7", "qwen14", "phimini", "phismall"
))

model_display <- c(
  human = "Human",
  gpt4omini = "GPT-4o-mini",
  gpt52 = "GPT-5.2",
  qwen7 = "Qwen2.5-7B-Instruct",
  qwen32 = "Qwen2.5-32B",
  deepseek = "DeepSeek-3.2",
  grok3 = "Grok-3",
  llama = "Llama-3.1-8B",
  qwen14 = "Qwen2.5-14B-Instruct",
  phimini = "Phi-3-mini-4k-instruct",
  phismall = "Phi-3-small-8k-instruct",
  phimed = "Phi-3-medium"
)

human_fmi <- fmi_summary$FMI[fmi_summary$model_key == "human"]

make_fmi_plot <- function(keys, compact = FALSE) {
  keys <- unique(c("human", keys))
  dat <- fmi_summary %>%
    filter(model_key %in% keys) %>%
    mutate(
      model = unname(model_display[model_key]),
      model = factor(model, levels = rev(unname(model_display[keys]))),
      value_label = sprintf("%.2f", FMI)
    )

  ggplot(dat, aes(FMI, model)) +
    geom_vline(xintercept = human_fmi, linetype = "dashed", linewidth = .25,
               colour = "#6F88AA", alpha = .80) +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high, colour = model_key == "human"), orientation = "y", width = 0,
                  linewidth = .42) +
    geom_point(aes(fill = model_key == "human"), shape = 21, size = if (compact) 2.0 else 2.60, stroke = .3,
               colour = "white") +
    geom_text(aes(x = ci_high + .012, label = value_label, colour = model_key == "human"), hjust = 0,
              family = base_family, size = if (compact) 1.75 else 2.28) +
    scale_colour_manual(values = c("FALSE" = "#2F6FB0", "TRUE" = "#2F6FB0"), guide = "none") +
    scale_fill_manual(values = c("FALSE" = "#2F6FB0", "TRUE" = "#2F6FB0"), guide = "none") +
    scale_x_continuous(limits = c(.35, 1.055), breaks = c(.4, .6, .8, 1.0), expand = c(0, 0)) +
    scale_y_discrete(expand = expansion(add = c(.25, .25))) +
    labs(x = "Factor Modularity Index (FMI)", y = NULL) +
    theme_classic(base_family = base_family, base_size = 6) +
    theme(
      panel.grid.major.x = element_line(colour = "#ECEFF2", linewidth = .2),
      axis.text.y = element_text(size = if (compact) 4.9 else 6.4, colour = "#303030"),
      axis.text.x = element_text(size = if (compact) 5.0 else 5.8, colour = "#4A4A4A"),
      axis.title.x = element_text(size = if (compact) 5.3 else 6.7, face = "bold",
                                  colour = "#303030", margin = margin(t = 2)),
      axis.line = element_line(linewidth = .3, colour = "#454545"),
      axis.ticks = element_line(linewidth = .3, colour = "#454545"),
      plot.margin = margin(0, 2, 1, 1)
    )
}

fmi_panel <- make_fmi_plot(primary_models)
panel_a <- wrap_plots(
  list(top_grid, legend_strip),
  ncol = 1,
  heights = c(.915, .085)
) + plot_annotation(
  title = "a",
  theme = theme(plot.title = element_text(family = base_family, face = "bold",
                                          size = 8, colour = "#171717",
                                          margin = margin(b = 0)))
)
panel_b <- fmi_panel + plot_annotation(
  title = "b",
  theme = theme(plot.title = element_text(family = base_family, face = "bold",
                                          size = 8, colour = "#171717",
                                          margin = margin(b = 0)))
)
divider <- ggplot() +
  geom_segment(aes(x = 0, xend = 1, y = .5, yend = .5), linewidth = .22, colour = "#D5D5D5") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  theme_void()
figure <- wrap_plots(
  list(wrap_elements(full = panel_a), free(divider), wrap_elements(full = panel_b)),
  ncol = 1,
  heights = c(.79, .001, .209)
) & theme(plot.background = element_rect(fill = "white", colour = NA))

supplementary_figure <- wrap_plots(
  list(make_fmi_plot(supplementary_models, compact = TRUE)), ncol = 1
) + plot_annotation(
  title = "Factor Modularity Index",
  theme = theme(plot.title = element_text(family = base_family, face = "bold",
                                          size = 8, colour = "#171717"))
) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

pdf_path <- file.path(output_dir, "figure3_centered_diamonds_phi_medium_ordered.pdf")
svg_path <- file.path(output_dir, "figure3_centered_diamonds_phi_medium_ordered.svg")
png_path <- file.path(output_dir, "figure3_centered_diamonds_phi_medium_ordered.png")
tiff_path <- file.path(output_dir, "figure3_centered_diamonds_phi_medium_ordered.tiff")
raster_stage_dir <- output_dir
dir.create(raster_stage_dir, recursive = TRUE, showWarnings = FALSE)
png_stage <- file.path(raster_stage_dir, "figure3_centered_diamonds_phi_medium_ordered.png")
tiff_stage <- file.path(raster_stage_dir, "figure3_centered_diamonds_phi_medium_ordered.tiff")
supp_pdf_path <- file.path(output_dir, "supplementary_factor_modularity_centered_all_models.pdf")
supp_svg_path <- file.path(output_dir, "supplementary_factor_modularity_centered_all_models.svg")
supp_png_stage <- file.path(raster_stage_dir, "supplementary_factor_modularity_centered_all_models.png")
supp_tiff_stage <- file.path(raster_stage_dir, "supplementary_factor_modularity_centered_all_models.tiff")

ggsave(pdf_path, figure, width = 183, height = 121, units = "mm", device = cairo_pdf)
ggsave(svg_path, figure, width = 183, height = 121, units = "mm", device = svglite::svglite)
ggsave(png_stage, figure, width = 183, height = 121, units = "mm", dpi = 300, device = ragg::agg_png)
ggsave(tiff_stage, figure, width = 183, height = 121, units = "mm", dpi = 600,
       device = ragg::agg_tiff, compression = "lzw")
ggsave(supp_pdf_path, supplementary_figure, width = 120, height = 110, units = "mm", device = cairo_pdf)
ggsave(supp_svg_path, supplementary_figure, width = 120, height = 110, units = "mm", device = svglite::svglite)
ggsave(supp_png_stage, supplementary_figure, width = 120, height = 110, units = "mm",
       dpi = 300, device = ragg::agg_png)
ggsave(supp_tiff_stage, supplementary_figure, width = 120, height = 110, units = "mm",
       dpi = 600, device = ragg::agg_tiff, compression = "lzw")

print(fmi_summary %>% arrange(desc(FMI)))
message("Saved: ", pdf_path)
message("Saved: ", svg_path)
message("Saved raster staging file: ", png_stage)
message("Saved raster staging file: ", tiff_stage)
message("Saved: ", supp_pdf_path)
message("Saved: ", supp_svg_path)
message("Saved raster staging file: ", supp_png_stage)
message("Saved raster staging file: ", supp_tiff_stage)
