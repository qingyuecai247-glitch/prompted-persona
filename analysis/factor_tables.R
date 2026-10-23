#!/usr/bin/env Rscript

# Recompute Supplementary factor-analysis summaries from the same centred
# person-level SCSC matrices used for the current main-text Figure 3.

# Use the R library configured by the local environment.

suppressPackageStartupMessages({
  library(psych)
  library(Matrix)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/")
project_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/")
result_dir <- file.path(project_dir, "analysis", "results",
                        "factor_modularity_centered_diamonds_phi_medium")

model_keys <- c("human", "deepseek", "grok3", "gpt52", "qwen32", "phimed",
                "gpt4omini", "llama", "qwen7", "qwen14", "phimini", "phismall")
model_labels <- c(
  human = "Human", deepseek = "DeepSeek-3.2", grok3 = "Grok-3",
  gpt52 = "GPT-5.2", qwen32 = "Qwen2.5-32B-Instruct",
  phimed = "Phi-3-medium-4k-instruct", gpt4omini = "GPT-4o-mini",
  llama = "Llama-3.1-8B-Instruct", qwen7 = "Qwen2.5-7B-Instruct",
  qwen14 = "Qwen2.5-14B-Instruct", phimini = "Phi-3-mini-4k-instruct",
  phismall = "Phi-3-small-8k-instruct"
)

nearest_cor <- function(x) {
  x <- (x + t(x)) / 2
  diag(x) <- 1
  if (min(eigen(x, symmetric = TRUE, only.values = TRUE)$values) < 1e-8) {
    x <- as.matrix(nearPD(x, corr = TRUE, keepDiag = TRUE)$mat)
  }
  x
}

read_matrix <- function(key) {
  path <- file.path(result_dir, paste0("person_level_scsc_centered_", key, ".csv"))
  dat <- read.csv(path, check.names = FALSE)
  x <- dat[, setdiff(names(dat), "id_p"), drop = FALSE]
  x[] <- lapply(x, as.numeric)
  as.data.frame(x)
}

fit_minres <- function(x, nfactors = 5L) {
  r <- nearest_cor(cor(x, use = "pairwise.complete.obs"))
  suppressWarnings(psych::fa(r, nfactors = nfactors, rotate = "oblimin",
                             fm = "minres", warnings = FALSE))
}

# Three- to six-factor solution summaries.
solution_rows <- list()
for (key in model_keys) {
  x <- read_matrix(key)
  for (nf in 3:6) {
    fit <- fit_minres(x, nf)
    ss <- sum(colSums(unclass(fit$loadings)^2))
    solution_rows[[length(solution_rows) + 1L]] <- data.frame(
      model_key = key, model = unname(model_labels[key]), n_factors = nf,
      total_ss_loadings = ss, proportion_variance = ss / ncol(x),
      cumulative_variance = ss / ncol(x), RMSR = fit$crms
    )
  }
}
write.csv(do.call(rbind, solution_rows),
          file.path(result_dir, "factor_solution_3_to_6_centered.csv"), row.names = FALSE)

# FMI threshold sensitivity from the exact five-factor loadings used in Fig. 3.
loadings <- read.csv(file.path(result_dir, "all_factor_loadings.csv"),
                     stringsAsFactors = FALSE)
fmi_at <- function(dat, threshold) {
  salient <- dat[abs(dat$loading) > threshold, , drop = FALSE]
  if (!nrow(salient)) return(NA_real_)
  factors <- unique(salient$factor)
  modularity <- vapply(factors, function(f) {
    tab <- table(salient$trait[salient$factor == f])
    max(tab) / sum(tab)
  }, numeric(1))
  mean(modularity)
}
fmi_rows <- do.call(rbind, lapply(model_keys, function(key) {
  dat <- loadings[loadings$model_key == key, , drop = FALSE]
  data.frame(model_key = key, model = unname(model_labels[key]),
             loading_threshold = c(.4, .5, .6),
             FMI = vapply(c(.4, .5, .6), function(z) fmi_at(dat, z), numeric(1)))
}))
write.csv(fmi_rows, file.path(result_dir, "fmi_threshold_sensitivity_centered.csv"),
          row.names = FALSE)

# Split-half factor congruence. Factors are matched across halves by maximizing
# the mean absolute Tucker coefficient over all 5! permutations.
all_permutations <- function(x) {
  if (length(x) == 1L) return(matrix(x, nrow = 1L))
  do.call(rbind, lapply(seq_along(x), function(i) cbind(x[i], all_permutations(x[-i]))))
}
perms <- all_permutations(1:5)
set.seed(20260902)
split_rows <- list()
for (key in model_keys) {
  x <- read_matrix(key)
  vals <- rep(NA_real_, 100)
  for (rep in seq_len(100)) {
    idx <- sample.int(nrow(x), floor(nrow(x) / 2))
    fit1 <- try(fit_minres(x[idx, , drop = FALSE], 5), silent = TRUE)
    fit2 <- try(fit_minres(x[-idx, , drop = FALSE], 5), silent = TRUE)
    if (inherits(fit1, "try-error") || inherits(fit2, "try-error")) next
    l1 <- unclass(fit1$loadings)
    l2 <- unclass(fit2$loadings)
    cong <- psych::factor.congruence(l1, l2, digits = 10)
    scores <- apply(perms, 1, function(p) mean(abs(cong[cbind(1:5, p)]), na.rm = TRUE))
    vals[rep] <- scores[which.max(scores)]
  }
  valid <- vals[is.finite(vals)]
  split_rows[[length(split_rows) + 1L]] <- data.frame(
    model_key = key, model = unname(model_labels[key]), splits_requested = 100,
    valid_splits = length(valid), mean_tucker = mean(valid), sd_tucker = sd(valid),
    min_tucker = min(valid)
  )
}
write.csv(do.call(rbind, split_rows),
          file.path(result_dir, "split_half_tucker_centered.csv"), row.names = FALSE)

message("Recomputed centred factor appendix statistics in: ", result_dir)
