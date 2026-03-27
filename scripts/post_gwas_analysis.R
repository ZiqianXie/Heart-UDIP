#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(ggpubr)
  library(CMplot)
  library(VennDiagram)
  library(grid)
  library(stringi)
  library(UpSetR)
})

usage <- function() {
  cat(
    paste(
      "Usage:",
      "  Rscript scripts/post_gwas_analysis.R --task=<task> [--key=value ...]",
      "  Rscript scripts/post_gwas_analysis.R --config=configs/post_gwas_4ch.yaml [--key=value ...]",
      "",
      "Tasks:",
      "  heart-manhattan      Plot GWAS Manhattan plot with lead-SNP annotation and optional Venn comparison",
      "  overlap-summary      Compare two lead-SNP result sets and summarize overlap/novelty",
      "  trait-summary        Summarize GWAS Catalog traits and export table + plot",
      "  function-enrichment  Plot MAGMA functional enrichment",
      "  disease-enrichment   Plot disease enrichment",
      "  tissue-enrichment    Plot tissue enrichment",
      "  celltype-enrichment  Plot cell-type enrichment",
      "  magma-manhattan      Plot gene-level MAGMA Manhattan plot",
      "  gene-overlap         Compare MAGMA-significant genes and mapped genes across two views",
      "  upset                Draw an UpSet plot from multiple set files",
      "",
      "Examples:",
      "  Rscript scripts/post_gwas_analysis.R --task=heart-manhattan --gwas-file=results/gwas/4ch.tsv --annotation-dir=results/fuma/4ch --output-prefix=results/post_gwas/4ch",
      "  Rscript scripts/post_gwas_analysis.R --config=configs/post_gwas_4ch.yaml",
      "  Rscript scripts/post_gwas_analysis.R --config=configs/post_gwas_4ch.csv --output-prefix=results/post_gwas/4ch_override",
      "  Rscript scripts/post_gwas_analysis.R --task=overlap-summary --lead-file-a=results/fuma/4ch/leadSNPs.txt --lead-file-b=results/fuma/2ch/leadSNPs.txt --output-prefix=results/post_gwas/overlap",
      "  Rscript scripts/post_gwas_analysis.R --task=function-enrichment --input-file=results/fuma/4ch/magma.gsa.out --output-file=results/post_gwas/4ch_function_enrichment.pdf",
      sep = "\n"
    )
  )
}

parse_cli_kv <- function(args) {
  parsed <- list()
  for (arg in args) {
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0)
    }
    if (!startsWith(arg, "--")) {
      stop("Arguments must use --key=value format: ", arg)
    }
    key_value <- sub("^--", "", arg)
    parts <- strsplit(key_value, "=", fixed = TRUE)[[1]]
    key <- parts[1]
    value <- if (length(parts) > 1) paste(parts[-1], collapse = "=") else "TRUE"
    parsed[[key]] <- value
  }
  parsed
}

trim_ws <- function(x) {
  sub("^\\s+", "", sub("\\s+$", "", x))
}

strip_wrapping_quotes <- function(x) {
  x <- trim_ws(x)
  if (nchar(x) >= 2) {
    first <- substr(x, 1, 1)
    last <- substr(x, nchar(x), nchar(x))
    if ((first == "\"" && last == "\"") || (first == "'" && last == "'")) {
      return(substr(x, 2, nchar(x) - 1))
    }
  }
  x
}

read_yaml_config <- function(path) {
  lines <- readLines(path, warn = FALSE)
  parsed <- list()
  for (line in lines) {
    stripped <- trim_ws(line)
    if (identical(stripped, "") || startsWith(stripped, "#")) {
      next
    }
    no_comment <- sub("\\s+#.*$", "", stripped)
    parts <- strsplit(no_comment, ":", fixed = TRUE)[[1]]
    if (length(parts) < 2) {
      stop("Invalid YAML config line: ", line)
    }
    key <- trim_ws(parts[1])
    value <- strip_wrapping_quotes(paste(parts[-1], collapse = ":"))
    parsed[[key]] <- value
  }
  parsed
}

read_csv_config <- function(path) {
  df <- fread(path, data.table = FALSE)
  key_col <- pick_column(df, c("key", "name", "parameter"), "Config key column")
  value_col <- pick_column(df, c("value", "val"), "Config value column")
  parsed <- as.list(as.character(df[[value_col]]))
  names(parsed) <- as.character(df[[key_col]])
  parsed
}

read_config_file <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("yaml", "yml")) {
    return(read_yaml_config(path))
  }
  if (ext == "csv") {
    return(read_csv_config(path))
  }
  stop("Unsupported config format: ", path, ". Use .yaml, .yml, or .csv")
}

merge_args <- function(base_args, override_args) {
  merged <- base_args
  for (name in names(override_args)) {
    merged[[name]] <- override_args[[name]]
  }
  merged
}

parse_args <- function() {
  cli_args <- parse_cli_kv(commandArgs(trailingOnly = TRUE))
  config_path <- cli_args[["config"]]
  if (is.null(config_path)) {
    return(cli_args)
  }

  config_args <- read_config_file(config_path)
  merged <- merge_args(config_args, cli_args)
  merged[["config"]] <- config_path
  merged
}

require_arg <- function(args, key) {
  value <- args[[key]]
  if (is.null(value) || identical(value, "")) {
    stop("Missing required argument --", key)
  }
  value
}

get_arg <- function(args, key, default = NULL) {
  value <- args[[key]]
  if (is.null(value) || identical(value, "")) default else value
}

as_integer_arg <- function(args, key, default = NULL) {
  value <- get_arg(args, key, default)
  if (is.null(value)) return(NULL)
  as.integer(value)
}

as_numeric_arg <- function(args, key, default = NULL) {
  value <- get_arg(args, key, default)
  if (is.null(value)) return(NULL)
  as.numeric(value)
}

as_bool_arg <- function(args, key, default = FALSE) {
  value <- tolower(as.character(get_arg(args, key, default)))
  value %in% c("true", "1", "yes", "y")
}

ensure_parent_dir <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

read_table_auto <- function(path, ...) {
  fread(path, data.table = FALSE, ...)
}

pick_column <- function(df, candidates, label) {
  match <- candidates[candidates %in% colnames(df)][1]
  if (is.na(match)) {
    stop(label, " not found. Tried: ", paste(candidates, collapse = ", "))
  }
  match
}

draw_fisher_venn <- function(set1, set2, background, name1, name2, output_file, p_cutoff = 0.05 / 39) {
  inter_ab <- length(intersect(set1, set2))
  only_a <- length(set1) - inter_ab
  only_b <- length(set2) - inter_ab
  neither_ab <- length(background) - length(union(set1, set2))

  fisher_mat <- matrix(c(inter_ab, only_a, only_b, neither_ab), nrow = 2, byrow = TRUE)
  fisher_result <- fisher.test(fisher_mat)
  if (fisher_result$p.value >= p_cutoff) {
    return(invisible(fisher_result))
  }

  ensure_parent_dir(output_file)
  png(output_file, width = 900, height = 900, res = 300)
  grid.newpage()
  venn_plot <- venn.diagram(
    x = setNames(list(set1, set2), c(name1, name2)),
    filename = NULL,
    fill = c("#66C2A5", "#FC8D62"),
    alpha = 0.5,
    cex = 0.7,
    cat.cex = 0.7,
    cat.fontface = "plain",
    cat.pos = c(-20, 20),
    margin = 0.2
  )
  grid.draw(venn_plot)
  grid.text(
    sprintf(
      "Fisher exact test:\np-value = %.3g\nOdds ratio = %.3g",
      fisher_result$p.value,
      as.numeric(fisher_result$estimate)
    ),
    x = unit(0.5, "npc"),
    y = unit(0.12, "npc"),
    gp = gpar(fontsize = 10)
  )
  dev.off()
  invisible(fisher_result)
}

plot_heart_manhattan <- function(args) {
  gwas_file <- require_arg(args, "gwas-file")
  annotation_dir <- require_arg(args, "annotation-dir")
  output_prefix <- require_arg(args, "output-prefix")

  gwas <- read_table_auto(gwas_file)
  lead_snp <- read_table_auto(file.path(annotation_dir, get_arg(args, "lead-file-name", "leadSNPs.txt")))
  snp_annotations <- read_table_auto(file.path(annotation_dir, get_arg(args, "snps-file-name", "snps.txt")))

  p_col <- pick_column(gwas, c("P", "p", "pvalue"), "GWAS p-value column")
  chr_col <- pick_column(gwas, c("CHR", "#CHROM", "chromosome"), "GWAS chromosome column")
  pos_col <- pick_column(gwas, c("POS", "BP", "base_pair_location"), "GWAS position column")
  snp_col <- pick_column(gwas, c("SNP", "rsid", "ID"), "GWAS SNP column")

  lead_id_col <- pick_column(lead_snp, c("rsID", "SNP", "Lead_SNP"), "Lead SNP identifier column")
  lead_p_col <- pick_column(lead_snp, c("p", "P", "leadP"), "Lead SNP p-value column")
  annot_id_col <- pick_column(snp_annotations, c("rsID", "SNP"), "Annotation SNP identifier column")
  annot_gene_col <- pick_column(snp_annotations, c("nearestGene", "GENE", "mappedGene"), "Annotation gene column")

  lead_snp <- lead_snp[order(as.numeric(lead_snp[[lead_p_col]])), , drop = FALSE]
  top_n <- as_integer_arg(args, "top-leads", 30)
  lead_snp <- head(lead_snp, top_n)
  snp_annotations <- snp_annotations[!duplicated(snp_annotations[[annot_id_col]]), , drop = FALSE]
  rownames(snp_annotations) <- snp_annotations[[annot_id_col]]
  matched_annotations <- snp_annotations[lead_snp[[lead_id_col]], , drop = FALSE]

  cmplot_df <- data.frame(
    SNP = gwas[[snp_col]],
    Chromosome = as.numeric(gwas[[chr_col]]),
    Position = as.numeric(gwas[[pos_col]]),
    P = as.numeric(gwas[[p_col]])
  )
  cmplot_df <- cmplot_df[is.finite(cmplot_df$P) & !is.na(cmplot_df$Chromosome) & !is.na(cmplot_df$Position), ]

  ensure_parent_dir(paste0(output_prefix, "_manhattan.pdf"))
  CMplot(
    cmplot_df,
    plot.type = "m",
    type = "p",
    LOG10 = TRUE,
    threshold = c(5e-8, 5e-8 / 256),
    highlight = lead_snp[[lead_id_col]],
    highlight.text = matched_annotations[[annot_gene_col]],
    highlight.text.cex = 1.2,
    highlight.col = "red",
    highlight.cex = 1.2,
    col = c("grey", "skyblue"),
    axis.cex = 1.2,
    lab.cex = 1.3,
    file = "pdf",
    file.name = paste0(output_prefix, "_manhattan"),
    dpi = 300
  )

  previous_heart <- get_arg(args, "previous-heart")
  previous_ecg <- get_arg(args, "previous-ecg")
  if (!is.null(previous_heart) && !is.null(previous_ecg)) {
    previous_heart_df <- read_table_auto(previous_heart)
    previous_ecg_df <- read_table_auto(previous_ecg)
    previous_heart_col <- pick_column(previous_heart_df, c("Lead SNV", "SNP", "rsID"), "Previous heart SNP column")
    previous_ecg_col <- pick_column(previous_ecg_df, c("SNP", "Lead SNV", "rsID"), "Previous ECG SNP column")
    venn.diagram(
      x = list(
        UDIP = unique(lead_snp[[lead_id_col]]),
        ECG = unique(previous_ecg_df[[previous_ecg_col]]),
        Shape = unique(previous_heart_df[[previous_heart_col]])
      ),
      category.names = c("UDIP", "ECG", "Shape"),
      filename = paste0(output_prefix, "_lead_snp_venn.png"),
      output = TRUE,
      imagetype = "png",
      height = 3000,
      width = 3000,
      resolution = 500,
      fill = c("#66c2a5", "#fc8d62", "#8da0cb"),
      alpha = 0.6,
      cex = 1.5,
      cat.cex = 1.5,
      cat.col = c("black", "black", "black"),
      lwd = 1.5,
      lty = "solid"
    )
  }
}

overlap_summary <- function(args) {
  lead_file_a <- require_arg(args, "lead-file-a")
  lead_file_b <- require_arg(args, "lead-file-b")
  output_prefix <- require_arg(args, "output-prefix")

  lead_a <- read_table_auto(lead_file_a)
  lead_b <- read_table_auto(lead_file_b)
  lead_col_a <- pick_column(lead_a, c("rsID", "SNP", "Lead_SNP"), "lead-file-a SNP column")
  lead_col_b <- pick_column(lead_b, c("rsID", "SNP", "Lead_SNP"), "lead-file-b SNP column")

  set_a <- unique(as.character(lead_a[[lead_col_a]]))
  set_b <- unique(as.character(lead_b[[lead_col_b]]))
  overlap <- intersect(set_a, set_b)
  only_a <- setdiff(set_a, set_b)
  only_b <- setdiff(set_b, set_a)

  summary_df <- data.frame(
    metric = c("n_a", "n_b", "n_overlap", "n_only_a", "n_only_b"),
    value = c(length(set_a), length(set_b), length(overlap), length(only_a), length(only_b))
  )
  ensure_parent_dir(paste0(output_prefix, "_summary.tsv"))
  write.table(summary_df, paste0(output_prefix, "_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(data.frame(snp = overlap), paste0(output_prefix, "_overlap.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(data.frame(snp = only_a), paste0(output_prefix, "_only_a.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(data.frame(snp = only_b), paste0(output_prefix, "_only_b.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

  venn.diagram(
    x = list(ViewA = set_a, ViewB = set_b),
    category.names = c(get_arg(args, "label-a", "ViewA"), get_arg(args, "label-b", "ViewB")),
    filename = paste0(output_prefix, "_venn.png"),
    output = TRUE,
    imagetype = "png",
    height = 2400,
    width = 2400,
    resolution = 400,
    fill = c("#66c2a5", "#fc8d62"),
    alpha = 0.6,
    cex = 1.4,
    cat.cex = 1.4,
    lwd = 1.5
  )
}

trait_summary <- function(args) {
  input_file <- require_arg(args, "input-file")
  output_prefix <- require_arg(args, "output-prefix")
  top_n <- as_integer_arg(args, "top-n", 25)

  trait_df <- read_table_auto(input_file)
  trait_col <- pick_column(trait_df, c("Trait", "trait", "DISEASE.TRAIT"), "Trait column")
  p_col <- pick_column(trait_df, c("P", "p", "pval"), "Trait p-value column")

  summary_df <- trait_df %>%
    mutate(.trait = .data[[trait_col]], .p = suppressWarnings(as.numeric(.data[[p_col]]))) %>%
    filter(!is.na(.trait)) %>%
    group_by(.trait) %>%
    summarise(
      freq = n(),
      min_p = min(.p, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(logp = -log10(min_p)) %>%
    arrange(desc(freq), desc(logp))

  write.csv(summary_df, paste0(output_prefix, "_trait_summary.csv"), row.names = FALSE)
  plot_df <- head(summary_df, top_n)
  plot_df$.trait <- factor(plot_df$.trait, levels = rev(plot_df$.trait))
  p <- ggplot(plot_df, aes(x = .trait, y = freq, fill = logp)) +
    geom_col() +
    coord_flip() +
    scale_fill_gradient(low = "grey80", high = "red") +
    labs(x = "Trait", y = "Frequency", fill = "-log10(min P)") +
    theme_minimal(base_size = 12)
  ggsave(paste0(output_prefix, "_trait_summary.pdf"), p, width = 10, height = 8)
}

plot_function_enrichment <- function(args) {
  input_file <- require_arg(args, "input-file")
  output_file <- require_arg(args, "output-file")
  top_n <- as_integer_arg(args, "top-n", 20)

  magma_functions <- fread(input_file, data.table = FALSE, skip = 4)
  p_col <- pick_column(magma_functions, c("P", "p"), "MAGMA enrichment p-value column")
  name_col <- pick_column(magma_functions, c("FULL_NAME", "Name"), "MAGMA term name column")

  magma_functions$FDR <- -log10(p.adjust(as.numeric(magma_functions[[p_col]]), method = "bonferroni"))
  magma_functions <- magma_functions[order(magma_functions$FDR, decreasing = TRUE), , drop = FALSE]
  magma_functions <- magma_functions[magma_functions$FDR > 1.3, , drop = FALSE]
  magma_functions <- head(magma_functions, top_n)
  magma_functions[[name_col]] <- sapply(magma_functions[[name_col]], function(x) {
    parts <- stringi::stri_split_fixed(x, "BP_", simplify = TRUE)
    if (ncol(parts) >= 2) parts[2] else x
  })
  magma_functions <- na.omit(magma_functions)
  magma_functions <- magma_functions[order(magma_functions$FDR), , drop = FALSE]
  magma_functions[[name_col]] <- tolower(magma_functions[[name_col]])

  p <- ggbarplot(magma_functions, x = name_col, y = "FDR", fill = "skyblue") +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", linewidth = 1) +
    xlab("GO Biological Process") +
    ylab("Enrichment (-log10 adjusted P)") +
    coord_flip() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 12)
    )

  ensure_parent_dir(output_file)
  ggsave(output_file, p, width = 11, height = 13)
  write.csv(magma_functions, sub("\\.pdf$", ".csv", output_file), row.names = FALSE)
}

plot_disease_enrichment <- function(args) {
  input_file <- require_arg(args, "input-file")
  output_file <- require_arg(args, "output-file")
  top_n <- as_integer_arg(args, "top-n", 25)

  disease_df <- fread(input_file, sep = "\t", fill = TRUE, data.table = FALSE)
  category_col <- pick_column(disease_df, c("Category", "category"), "Category column")
  q_col <- pick_column(disease_df, c("q-value Bonferroni", "q_value_bonferroni", "FDR"), "Adjusted q-value column")
  name_col <- pick_column(disease_df, c("Name", "Trait", "Description"), "Disease label column")

  disease_df <- disease_df[disease_df[[category_col]] == "Disease", , drop = FALSE]
  disease_df$FDR <- -log10(as.numeric(disease_df[[q_col]]))
  disease_df <- disease_df[order(disease_df$FDR, decreasing = TRUE), , drop = FALSE]
  disease_df <- head(disease_df, top_n)
  disease_df <- disease_df[order(disease_df$FDR, decreasing = FALSE), , drop = FALSE]

  p <- ggbarplot(disease_df, x = name_col, y = "FDR", fill = "skyblue") +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", linewidth = 1) +
    xlab("Phenotype") +
    ylab("Enrichment (-log10 adjusted P)") +
    coord_flip()

  ensure_parent_dir(output_file)
  ggsave(output_file, p, width = 9, height = 10)
}

plot_tissue_enrichment <- function(args) {
  input_file <- require_arg(args, "input-file")
  output_file <- require_arg(args, "output-file")
  skip_lines <- as_integer_arg(args, "skip-lines", 5)
  start_row <- as_integer_arg(args, "start-row", 24)
  end_row <- as_integer_arg(args, "end-row", 54)

  tissue <- fread(input_file, data.table = FALSE, fill = TRUE, skip = skip_lines)
  p_col <- pick_column(tissue, c("P", "p"), "Tissue enrichment p-value column")
  name_col <- pick_column(tissue, c("FULL_NAME", "Name"), "Tissue name column")

  tissue$FDR <- -log10(p.adjust(as.numeric(tissue[[p_col]]), method = "bonferroni"))
  tissue <- tissue[order(tissue$FDR), , drop = FALSE]
  tissue <- tissue[start_row:min(end_row, nrow(tissue)), , drop = FALSE]

  p <- ggbarplot(tissue, x = name_col, y = "FDR", fill = "skyblue") +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", linewidth = 1) +
    xlab("Tissue") +
    ylab("Enrichment (-log10 adjusted P)") +
    coord_flip() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 12)
    )

  ensure_parent_dir(output_file)
  ggsave(output_file, p, width = 11, height = 13)
}

plot_celltype_enrichment <- function(args) {
  input_file <- require_arg(args, "input-file")
  output_file <- require_arg(args, "output-file")
  exclude_blood <- as_bool_arg(args, "exclude-blood", TRUE)

  cell <- fread(input_file, data.table = FALSE)
  dataset_col <- pick_column(cell, c("Dataset", "dataset"), "Dataset column")
  celltype_col <- pick_column(cell, c("Cell_type", "celltype", "CellType"), "Cell type column")
  padj_col <- pick_column(cell, c("P.adj", "padj", "P_adj"), "Adjusted P column")
  beta_col <- pick_column(cell, c("BETA", "beta"), "Beta column")
  p_col <- pick_column(cell, c("P", "p"), "Raw P column")

  if (exclude_blood) {
    cell <- cell[!(cell[[dataset_col]] %in% c(
      "GSE89232_Human_Blood",
      "539_Travaglini_2020_Blood_level1",
      "541_Xu_Human_2023_Blood_level1"
    )), , drop = FALSE]
  }

  enrichment_df <- data.frame(
    celltype = names(tapply(cell[[padj_col]], cell[[celltype_col]], mean, na.rm = TRUE)),
    adjust_p = as.numeric(tapply(cell[[padj_col]], cell[[celltype_col]], mean, na.rm = TRUE)),
    beta = as.numeric(tapply(cell[[beta_col]], cell[[celltype_col]], mean, na.rm = TRUE)),
    p = as.numeric(tapply(cell[[p_col]], cell[[celltype_col]], mean, na.rm = TRUE))
  )
  enrichment_df$adjust_p <- -log10(enrichment_df$adjust_p)
  enrichment_df <- enrichment_df[order(enrichment_df$adjust_p), , drop = FALSE]

  p <- ggbarplot(enrichment_df, x = "celltype", y = "adjust_p", fill = "skyblue") +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", linewidth = 1) +
    xlab("Cell types") +
    ylab("Enrichment (-log10 adjusted P)") +
    coord_flip() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 12)
    )

  ensure_parent_dir(output_file)
  ggsave(output_file, p, width = 12, height = 6)
  write.csv(enrichment_df, sub("\\.pdf$", ".csv", output_file), row.names = FALSE)
}

plot_magma_manhattan <- function(args) {
  input_file <- require_arg(args, "input-file")
  output_prefix <- require_arg(args, "output-prefix")
  significance_level <- as_numeric_arg(args, "significance-level", 0.05)
  total_genes <- as_integer_arg(args, "total-genes", 19128)

  magma <- fread(input_file, data.table = FALSE)
  required_cols <- c("SYMBOL", "CHR", "START", "P")
  if (!all(required_cols %in% colnames(magma))) {
    stop("MAGMA output must contain columns: ", paste(required_cols, collapse = ", "))
  }

  magma <- magma %>%
    filter(!is.na(CHR), !is.na(START), !is.na(P))
  threshold <- significance_level / total_genes
  cmplot_df <- data.frame(
    SNP = magma$SYMBOL,
    Chromosome = magma$CHR,
    Position = magma$START,
    P.value = magma$P
  )
  significant_genes <- cmplot_df$SNP[cmplot_df$P.value < threshold]

  ensure_parent_dir(paste0(output_prefix, ".jpg"))
  CMplot(
    cmplot_df,
    plot.type = "m",
    LOG10 = TRUE,
    threshold = threshold,
    threshold.col = "red",
    threshold.lty = 1,
    chr.den.col = NULL,
    amplify = TRUE,
    highlight = significant_genes,
    highlight.col = "blue",
    highlight.cex = 1.2,
    highlight.text = significant_genes,
    highlight.text.cex = 0.7,
    signal.cex = 1.2,
    file.output = TRUE,
    file = "jpg",
    dpi = 300,
    file.name = output_prefix,
    verbose = TRUE
  )
  write.csv(magma[magma$P < threshold, , drop = FALSE], paste0(output_prefix, "_significant_genes.csv"), row.names = FALSE)
}

extract_rsids <- function(values) {
  values %>%
    gsub(" ", "", .) %>%
    gsub(":", ";", .) %>%
    paste(collapse = ";") %>%
    strsplit(";") %>%
    unlist() %>%
    unique() %>%
    sort()
}

gene_overlap <- function(args) {
  magma_a_file <- require_arg(args, "magma-a")
  magma_b_file <- require_arg(args, "magma-b")
  genes_a_file <- require_arg(args, "genes-a")
  genes_b_file <- require_arg(args, "genes-b")
  output_prefix <- require_arg(args, "output-prefix")
  significance_level <- as_numeric_arg(args, "significance-level", 0.05 / 19291)

  magma_a <- read_table_auto(magma_a_file)
  magma_b <- read_table_auto(magma_b_file)
  genes_a <- read_table_auto(genes_a_file)
  genes_b <- read_table_auto(genes_b_file)

  symbol_col_magma_a <- pick_column(magma_a, c("SYMBOL", "symbol"), "magma-a symbol column")
  symbol_col_magma_b <- pick_column(magma_b, c("SYMBOL", "symbol"), "magma-b symbol column")
  p_col_a <- pick_column(magma_a, c("P", "p"), "magma-a p-value column")
  p_col_b <- pick_column(magma_b, c("P", "p"), "magma-b p-value column")
  symbol_col_genes_a <- pick_column(genes_a, c("symbol", "SYMBOL"), "genes-a symbol column")
  symbol_col_genes_b <- pick_column(genes_b, c("symbol", "SYMBOL"), "genes-b symbol column")

  sig_magma_a <- unique(magma_a[[symbol_col_magma_a]][as.numeric(magma_a[[p_col_a]]) < significance_level])
  sig_magma_b <- unique(magma_b[[symbol_col_magma_b]][as.numeric(magma_b[[p_col_b]]) < significance_level])
  mapped_genes_a <- unique(genes_a[[symbol_col_genes_a]])
  mapped_genes_b <- unique(genes_b[[symbol_col_genes_b]])

  venn.diagram(
    x = list(
      MAGMA_A = sig_magma_a,
      MAGMA_B = sig_magma_b,
      Mapped_A = mapped_genes_a,
      Mapped_B = mapped_genes_b
    ),
    category.names = c(
      get_arg(args, "label-magma-a", "MAGMA_A"),
      get_arg(args, "label-magma-b", "MAGMA_B"),
      get_arg(args, "label-mapped-a", "Mapped_A"),
      get_arg(args, "label-mapped-b", "Mapped_B")
    ),
    filename = paste0(output_prefix, "_venn.png"),
    output = TRUE,
    imagetype = "png",
    height = 3000,
    width = 3500,
    resolution = 500,
    fill = c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"),
    alpha = 0.6,
    cex = 1.2,
    cat.cex = 0.9,
    cat.col = c("black", "black", "black", "black"),
    lwd = 1.5
  )

  summary_df <- data.frame(
    set_name = c("sig_magma_a", "sig_magma_b", "mapped_genes_a", "mapped_genes_b"),
    n = c(length(sig_magma_a), length(sig_magma_b), length(mapped_genes_a), length(mapped_genes_b))
  )
  write.table(summary_df, paste0(output_prefix, "_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

read_set_spec <- function(spec) {
  parts <- strsplit(spec, "=", fixed = TRUE)[[1]]
  if (length(parts) != 2) {
    stop("Set spec must be name=path:column, got: ", spec)
  }
  name <- parts[1]
  path_col <- strsplit(parts[2], ":", fixed = TRUE)[[1]]
  if (length(path_col) != 2) {
    stop("Set spec must be name=path:column, got: ", spec)
  }
  df <- read_table_auto(path_col[1])
  column <- path_col[2]
  if (!(column %in% colnames(df))) {
    stop("Column ", column, " not found in ", path_col[1])
  }
  list(name = name, values = unique(as.character(df[[column]])))
}

plot_upset_sets <- function(args) {
  specs_raw <- require_arg(args, "sets")
  output_file <- require_arg(args, "output-file")
  nsets <- as_integer_arg(args, "nsets", 8)
  nintersects <- as_integer_arg(args, "nintersects", 50)

  specs <- strsplit(specs_raw, ";", fixed = TRUE)[[1]]
  set_list <- lapply(specs, read_set_spec)
  names(set_list) <- vapply(set_list, function(x) x$name, character(1))
  set_values <- lapply(set_list, function(x) x$values)

  ensure_parent_dir(output_file)
  pdf(output_file, width = 12, height = 8)
  UpSetR::upset(
    UpSetR::fromList(set_values),
    nsets = nsets,
    nintersects = nintersects,
    order.by = "freq",
    mainbar.y.label = "Intersection size",
    sets.x.label = "Set size",
    keep.order = TRUE
  )
  dev.off()
}

run_task <- function(args) {
  task <- require_arg(args, "task")
  switch(
    task,
    "heart-manhattan" = plot_heart_manhattan(args),
    "overlap-summary" = overlap_summary(args),
    "trait-summary" = trait_summary(args),
    "function-enrichment" = plot_function_enrichment(args),
    "disease-enrichment" = plot_disease_enrichment(args),
    "tissue-enrichment" = plot_tissue_enrichment(args),
    "celltype-enrichment" = plot_celltype_enrichment(args),
    "magma-manhattan" = plot_magma_manhattan(args),
    "gene-overlap" = gene_overlap(args),
    "upset" = plot_upset_sets(args),
    stop("Unknown task: ", task)
  )
}

main <- function() {
  args <- parse_args()
  if (length(args) == 0) {
    usage()
    quit(save = "no", status = 1)
  }
  run_task(args)
}

main()
