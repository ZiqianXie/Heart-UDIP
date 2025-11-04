## 
library(dplyr)
library(data.table)
library(stringi)
library(stringr)
library(dplyr)
library(stringi)
library(stringr)
library(data.table)
library(ggplot2)
library(ggpubr)
library(CMplot)
library(grid)
library(VennDiagram)
#
#’ Draw a Venn of two gene sets, run Fisher’s test, and annotate p‑value & OR
#’
#’ @param set1       Character vector of genes in set 1
#’ @param set2       Character vector of genes in set 2
#’ @param background Character vector of all background genes
#’ @param name1      Label for set 1
#’ @param name2      Label for set 2
#’ @param filename   Output file (e.g. "venn.png")
fisherVennPlot <- function(set1, set2, background,
                           name1 = "Set1", name2 = "Set2",
                           filename = "venn.png") {
  # 1. Counts
  inter_ab   <- length(intersect(set1, set2))
  only_a     <- length(set1) - inter_ab
  only_b     <- length(set2) - inter_ab
  neither_ab <- length(background) - length(union(set1, set2))
  
  # 2. Fisher table
  fisher_mat <- matrix(
    c(inter_ab, only_a,
      only_b,  neither_ab),
    nrow = 2, byrow = TRUE
  )
  #print
  # 3. Test
  ft   <- fisher.test(fisher_mat)
  pval <- ft$p.value
  or   <- as.numeric(ft$estimate)
  if (pval < (0.05/39)) {
    message("Significant set: ", name1)
    print(pval)
    # 4. Open device
    png(filename, width = 900, height = 900, res = 300)
    grid.newpage()

    # 5. Draw (increase margin to make space at bottom)
    venn.plot <- venn.diagram(
      x            = setNames(list(set1, set2), c(name1, name2)),
      filename     = NULL,
      fill         = c("#66C2A5", "#FC8D62"),
      alpha        = 0.5,
      cex          = 0.7,    # count label size
      cat.cex      = 0.7,    # category label size
      cat.fontface = "plain",
      cat.pos      = c(-20, 20),
      margin       = 0.2     # push diagram up a bit
    )
    grid.draw(venn.plot)

    # 6. Annotate (move up to y = 0.12)
    annotation <- sprintf(
      "Fisher's exact test:\np-value = %.3g\nOdds ratio = %.3g",
      pval, or
    )
    grid.text(
      annotation,
      x = unit(0.5, "npc"), y = unit(0.12, "npc"),
      gp = gpar(fontsize = 10)
    )

    # 7. Close
    dev.off()
    
    message("Saved: ", filename)
  }
  

}
### write.table(heat_file,'HEART_4ich_inputation_end.txt',row.names = FALSE,quote = FALSE,sep='\t')

plot_heart_manhattan<-function(heart_gwas,annotation_dir,save_name){
  #annotation_dir<-'./4ch_3Duin'
  #heart_gwas<-'HEART_4ich_inputation_3Duin.txt'
  heat_file<-fread(heart_gwas,data.table = FALSE)
  heat_file$BP<-heat_file$POS
  lead_snp<-fread(paste(annotation_dir,'/leadSNPs.txt',sep = ""),data.table = FALSE)
  lead_snp<-lead_snp[order(lead_snp$p),]
  lead_snp<-lead_snp[1:30,]
  ##
  gene_SNP<-fread(paste(annotation_dir,'/snps.txt',sep = ""),data.table = FALSE)
  gene_SNP<-gene_SNP[!duplicated(gene_SNP$rsID),]
  rownames(gene_SNP)<-gene_SNP$rsID
  gene_SNP<-gene_SNP[lead_snp$rsID,]
  ##
  #lead_snp<-lead_snp
  #heat_file_sig<-heat_file[which(heat_file$P<(5e-8)),]
  plot_heart<-heat_file[,c(2,1,9,8)]
  plot_heart<-plot_heart[which(plot_heart$P<0.5),]
  ## 1.482197e-323
  save_pdf<-paste(save_name,'.pdf',sep = "")
  CMplot(plot_heart,
         plot.type = "m",
         type = "p",
         LOG10 = TRUE,
         threshold = c(5e-8,5e-8/256),
         highlight.text=gene_SNP$nearestGene,
         highlight.text.cex =1.8,
         axis.cex = 1.5,
         lab.cex = 1.8,
       #  highlight.text.font=10,
         col = c("grey", "skyblue"),  # Alternating colors for different chromosomes
         highlight = lead_snp$rsID,  # Manually specify SNPs to highlight
         highlight.col = "red",       # Color for highlighted SNPs
         highlight.cex = 1.5,         # Point size for highlighted SNPs
         file = "pdf",
         file.name=save_pdf,
         dpi = 300)
  #####
  previous_discover<-fread('../heart_previous.csv',data.table = FALSE)
  previous_shape_discover_SNP<-previous_discover$`Lead SNV`
  ##
  previous_ECG<-fread('../ECG_previous.csv',data.table = FALSE)
  previous_ECG_SNP<-previous_ECG$SNP%>%unique()
  ## veen plot ##
    #save_name<-paste(annotation_dir,)
  venn_list<-list(UDIP=lead_snp$rsID,ECG=previous_ECG_SNP,shape=previous_shape_discover_SNP)
  save_name<-paste(save_name,'.png',sep = "")
  venn.plot <- venn.diagram(
    x = venn_list,
    category.names = c("UDIP", "ECG", "Shape"),
    filename = save_name, 
    output = TRUE,
    imagetype = "png",
    height = 3000,
    width = 3000,
    resolution = 500,
    fill = c("#66c2a5", "#fc8d62", "#8da0cb"),  # color
    alpha = 0.6,
    cex = 1.5,  # 文字大小
    cat.cex = 1.5,
    cat.col = c("black", "black", "black"),
    lwd = 1.5,
    lty = "solid"
  )
}
plot_heart_manhattan('HEART_4ich_final_inputation_3Duin.txt','./4ch_3Duin','4h_3D_uin')

plot_heart_manhattan('HEART_2ich_final_inputation_3Duin.txt','./2ch_3Duin/','2ch_3Duin')

#### 4 GWAS overlap ##
res_4ch_unet<-fread('4ch_3Duin/leadSNPs.txt',data.table = FALSE)
res__2ch_unet<-fread('2ch_3Duin/leadSNPs.txt',data.table = FALSE)
##  ##
res_gene_4ch_unet<-fread('4ch_3Duin/GenomicRiskLoci.txt',data.table = FALSE)
res_gene_2ch_unet<-fread('2ch_3Duin/GenomicRiskLoci.txt',data.table = FALSE)
###

intersect(res_4ch_unet$rsID,res_2ch_unet$rsID)%>%length()

###
venn_list<-list(res_4ch_gan=res_4ch_gan$rsID,res_4ch_unet=res_4ch_unet$rsID,
                res_2ch_gan=res_2ch_gan$rsID,res_2ch_unet=res_2ch_unet$rsID)
save_name<-paste('all_res_overgage_snp','.png',sep = "")
venn.plot <- venn.diagram(
  x = venn_list,
  category.names = c("4ch_3DGAN", "4ch_3Duin", "2ch_3DGAN","2ch_3Duin"),
  filename = save_name, 
  output = TRUE,
  imagetype = "png",
  height = 3000,
  width = 3000,
  resolution = 500,
  fill = c("#66c2a5", "#fc8d62", "#8da0cb",'#f7fb84'),  # color
  alpha = 0.6,
  cex = 1,  # 文字大小
  cat.cex = 1,
  cat.col = c("black", "black", "black","black"),
  lwd = 1.5,
  lty = "solid"
)
###  heart  SNP analysis 
catelog_4D_3Duin<-fread('./4ch_3Duin/gwascatalog.txt',data.table = FALSE) ##
catelog_2D_3Duin<-fread('./2ch_3Duin/gwascatalog.txt',data.table = FALSE) ##
##

##
##
word_df <- catelog_4D_3DGAN %>%
  mutate(pval = as.numeric(P)) %>%      
  group_by(Trait) %>%
  summarise(
    freq = n(),                             
    min_p = min(pval, na.rm = TRUE)         
  ) %>%
  mutate(logp = -log10(min_p))
library(ggwordcloud)
library(ggplot2)
word_df<-word_df[order(word_df$freq,decreasing = TRUE),]
word_df<-word_df[1:50,]
ggplot(word_df, aes(label = Trait, size = freq, color = logp)) +
  geom_text_wordcloud_area(family = "sans") +
  scale_size_area(max_size = 30) +
  scale_color_gradient(low = "grey80", high = "red") +
  theme_minimal()
ggplot(word_df, aes(label = Trait, size = freq, color = logp)) +
  geom_text_wordcloud_area(family = "sans") +
  scale_size_area(max_size = 40, name = "Frequency") +   # ← 字体大小含义
  scale_color_gradient(low = "grey80", high = "red", name = "-log10(P)") +  # ← 颜色含义
  labs(title = "Trait Word Cloud", subtitle = "Font size ~ Frequency; Color ~ Significance") +
  theme_minimal()
### preivous 
view_2ch_4ch_uni<-union(res_4ch_gan$rsID,res_2ch_gan$rsID)
previous_data_snp_all<-rbind(catelog_4D_3DGAN,catelog_2D_3DGAN)
previous_data_snp<-intersect(previous_data_snp_all$snp,view_2ch_4ch_uni) ## 111 
others_snp<-setdiff(view_2ch_4ch_uni,previous_data_snp_all$snp)
###
view_2ch_4ch_combined<-rbind(res_4ch_gan,res_2ch_gan)
view_2ch_4ch_combined<-view_2ch_4ch_combined[!duplicated(view_2ch_4ch_combined$rsID),]
rownames(view_2ch_4ch_combined)<-view_2ch_4ch_combined$rsID
overlap_2ch_4ch<-intersect(res_4ch_gan$rsID,res_2ch_gan$rsID)
specific_2ch<-setdiff(res_2ch_gan$rsID,res_4ch_gan$rsID)
specific_4ch<-setdiff(res_4ch_gan$rsID,res_2ch_gan$rsID)
phenotype_list<-c(rep('2Ch',length(specific_2ch)),rep('4Ch',length(specific_4ch)),
                  rep('Overlap(2Ch_4Ch)',length(overlap_2ch_4ch)))
view_2ch_4ch_combined<-view_2ch_4ch_combined[c(specific_2ch,specific_4ch,overlap_2ch_4ch),]
view_2ch_4ch_combined$phenotype<-phenotype_list
annotation_type<-c(rep('Previous',length(previous_data_snp)),rep('Novel',length(others_snp)))
view_2ch_4ch_combined<-view_2ch_4ch_combined[c(previous_data_snp,others_snp),]
###
intersect(catelog_4D_3Duin$IndSigSNP,res_4ch_unet$rsID)%>%unique()%>%length()
intersect(catelog_2D_3Duin$IndSigSNP,res_2ch_unet$rsID)%>%unique()%>%length()
view_2ch_4ch_combined$annotation<-annotation_type
####
view_2ch_4ch_combined<-view_2ch_4ch_combined[,c(4,5,6,10,11)]
colnames(view_2ch_4ch_combined)<-c('snp','chr','pos','phenotype','annotation')

###
rownames(res_4ch_gan)<-res_4ch_gan$rsID
previous_data<-res_4ch_gan[previous_data_snp,c(4,5,6)]
now_loci_data<-res_4ch_gan[setdiff(res_4ch_gan$rsID,previous_data_snp),c(4,5,6)]
combine_res<-rbind(previous_data,now_loci_data)
phenotype<-c(rep('Previous reported',nrow(previous_data)),rep('New identified',nrow(now_loci_data)))
combine_res$phenotype<-phenotype
colnames(combine_res)<-c('snp','chr','pos','phenotype')
##
write.table(view_2ch_4ch_combined,'2ch_4ch_3D_GAN_plot_lead_prevous.txt',row.names = FALSE,quote = FALSE,sep = '\t')
## common analysis 


# heart gene #
heart_gene<-fread('./4ch_3Duin/genes.txt',data.table = FALSE)
heart_gene1<-fread('./2ch_3Duin/genes.txt',data.table = FALSE)
###
test_gene<-union(heart_gene$symbol,heart_gene1$symbol)
LV_heart<-fread('UPE_LVgenes.txt',data.table = FALSE,header = FALSE)
intersect(LV_heart$V1,heart_gene$symbol)%>%length()
intersect(LV_heart$V1,test_gene)%>%length()

write.table(heart_gene$ensg,'4Ch_3Duin_heart_gene.txt',row.names = FALSE,quote = FALSE,col.names = FALSE)
write.table(heart_gene1$ensg,'2Ch_3Duin_heart_gene.txt',row.names = FALSE,quote = FALSE,col.names = FALSE)
## functional analysis   ##
plot_function_enrichment <- function(magma_file, output_file = "function_enrichment_plot.pdf") {
  
  magma_functions <- fread(magma_file, data.table = FALSE, skip = 4)
  magma_functions$FDR <- -log10(p.adjust(magma_functions$P, method = 'bonferroni'))
  magma_functions <- magma_functions[order(magma_functions$FDR, decreasing = TRUE), ]
  magma_functions <- magma_functions[magma_functions$FDR > 1.3, ]
  top_30 <- magma_functions[1:20,]
  print(nrow(magma_functions))
    
  top_30 <- top_30 %>%
    mutate(FULL_NAME = sapply(FULL_NAME, function(x) {
      parts <- stringi::stri_split_fixed(x, 'BP_', simplify = TRUE)
      if (ncol(parts) >= 2) parts[2] else NA
    }))
  top_30<-na.omit(top_30)  
  top_30<-top_30[order(top_30$FDR),]
  top_30$FULL_NAME<-tolower(top_30$FULL_NAME)
 # top_30<-top_30[1:10,]
  p <- ggbarplot(top_30, x = 'FULL_NAME', y = 'FDR', fill = 'skyblue') +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", size = 1) +
    xlab('GO Biological Process') + ylab('Enrichment (-log10 Adjusted P)') +
    coord_flip()+
    theme(
      axis.text.x = element_text(size = 14), 
      axis.text.y = element_text(size = 14),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14)
    )
  
  ggsave(output_file, p, width = 11, height = 13)
  return(magma_functions)
}
plot_disease_enrichment <- function(enrichment_file, output_file = "disease_enrichment_plot.pdf") {

  top_geneE <- fread(enrichment_file, sep = '\t', fill = TRUE, data.table = FALSE)
  disease <- top_geneE[top_geneE$Category == 'Disease', ]
  disease$FDR <- -log10(disease$`q-value Bonferroni`)
  disease <-disease[order(disease$FDR, decreasing = TRUE), ]
  disease=disease[1:25,]
  disease=disease[order(disease$FDR, decreasing = FALSE), ]
  p <- ggbarplot(disease, x = 'Name', y = 'FDR', fill = 'skyblue') +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", size = 1) +
    xlab('Phenotype') + ylab('Enrichment (-log10 Adjusted P)') +
    coord_flip()
  ggsave(output_file, p, width = 9, height = 10)
}
plot_tissue_enrichment <- function(tissue_file, output_file = "tissue_enrichment_plot.pdf", skip_lines = 5) {
  
  #tissue_file<-'./2ch_3Duin/magma_exp_gtex_v8_ts_general_avg_log2TPM.gsa.out'
  tissue <- fread(tissue_file, data.table = FALSE, fill = TRUE, skip = skip_lines)
  print(nrow(tissue))
  tissue$FDR <- -log10(p.adjust(tissue$P, method = 'bonferroni'))
  tissue <- tissue[order(tissue$FDR), ]
  n_tissue<-nrow(tissue)
  tissue<-tissue[24:54,]
  #tissue<-tissue[which(tissue$FDR>0),]
  p <- ggbarplot(tissue, x = 'FULL_NAME', y = 'FDR', fill = 'skyblue') +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", size = 1) +
    xlab('Tissue') + ylab('Enrichment (-log10 Adjusted P)') +
    theme(axis.text.x = element_text(angle = 0, hjust = 1))+coord_flip()+
    theme(
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14)
    )
  print(p)

  ggsave(output_file, p, width = 11, height = 13)
  return(tissue)
}
plot_celltype_enrichment <- function(cell_file, output_file = "celltype_enrichment_plot.pdf", exclude_blood = TRUE) {
  
 # cell_file<-'./2Ch_celltype/magma_celltype_step1.txt'
  cell <- fread(cell_file, data.table = FALSE)
  
  if (exclude_blood) {
    cell <- cell[!(cell$Dataset %in% c(
      "GSE89232_Human_Blood", 
      "539_Travaglini_2020_Blood_level1", 
      "541_Xu_Human_2023_Blood_level1")), ]
  }
  
  enrichment_df <- tapply(cell$P.adj, cell$Cell_type, mean) %>% as.data.frame()
  enrichment_beta <- tapply(cell$BETA, cell$Cell_type, mean) %>% as.data.frame()
  enrichment_P <- tapply(cell$P, cell$Cell_type, mean) %>% as.data.frame()
  enrichment_df$celltype <- rownames(enrichment_df)
  enrichment_df$beta<-enrichment_beta
  enrichment_df$P<-enrichment_P
  colnames(enrichment_df) <- c('adjust_p', 'celltype','BETA','P')
  enrichment_df$adjust_p <- -log10(enrichment_df$adjust_p)
  enrichment_df <- enrichment_df[order(enrichment_df$adjust_p), ]
  
  p <- ggbarplot(enrichment_df, x = 'celltype', y = 'adjust_p', fill = 'skyblue') +
    geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", size = 1) +
    xlab('Cell Types') + ylab('Enrichment (-log10 Adjusted P)') +
    coord_flip() +
    theme(
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14)
    )
  print(p)
  ggsave(output_file, p, width = 12, height = 6)
  return(enrichment_df)
}
# 4ch functional analysis #
view_4ch_function<-plot_function_enrichment('./4ch_3Duin/magma.gsa.out',output_file ='4ch_functional_enrichment.pdf')
view_2ch_function<-plot_function_enrichment('./2ch_3Duin/magma.gsa.out',output_file ='2ch_functional_enrichment.pdf')
##
write.csv(view_4ch_function,'4Ch_function.csv',row.names = FALSE)
write.csv(view_2ch_function,'2Ch_function.csv',row.names = FALSE)
##
plot_disease_enrichment('./2Ch_Top_gene_functions.txt',output_file ='2ch_phenotype_enrichment.pdf' )
tissue_4ch<-plot_disease_enrichment('./4Ch_Top_gene_functions.txt',output_file ='4ch_phenotype_enrichment.pdf' )
tissue_4ch<-plot_tissue_enrichment('./4ch_3Duin/magma_exp_gtex_v8_ts_avg_log2TPM.gsa.out',output_file ='4ch_tissue_enrichment.pdf' )
tissue_2ch<-plot_tissue_enrichment('./2ch_3Duin/magma_exp_gtex_v8_ts_avg_log2TPM.gsa.out',output_file ='2ch_tissue_enrichment.pdf' )
cell_2ch<-plot_celltype_enrichment('./2Ch_celltype/magma_celltype_step1.txt',output_file ='2ch_cell_type_enrichment.pdf' )
cell_4ch<-plot_celltype_enrichment('./4Ch_celltype/magma_celltype_step1.txt',output_file ='4ch_cell_type_enrichment.pdf' )
#
# Define a function to plot a Manhattan plot from MAGMA gene-level output
plot_magma_manhattan_with_labels <- function(magma_file, significance_level = 0.05, total_genes = 19128, output_name = "magma_manhattan") {
  # Load the MAGMA gene-level results
  #magma_file<-"./4ch_3Duin/magma.genes.out"
  magma_result <- fread(magma_file)
  
  # Check necessary columns exist
  required_cols <- c("SYMBOL", "CHR", "START", "P")
  if (!all(required_cols %in% colnames(magma_result))) {
    stop("MAGMA output must contain the following columns: GENE, CHR, START, and P")
  }
  
  # Remove missing or invalid data
  magma_result <- magma_result %>%
    filter(!is.na(CHR), !is.na(START), !is.na(P))
  
  # Compute genome-wide significance threshold
  threshold <- significance_level / total_genes
  
  # Prepare data for CMplot
  cmplot_data <- magma_result %>%
    mutate(SNP = SYMBOL,  # CMplot expects "SNP" column
           Chromosome = CHR,
           Position = START,
           P.value = P)
  cmplot_data<-cmplot_data[,c('SNP','Chromosome','Position','P.value')]
  # Identify significant genes (make sure they actually exist in the data)
  significant_genes <- cmplot_data %>%
    filter(P.value < threshold) %>%
    pull(SNP)
  
  # Confirm only genes that exist in cmplot_data are passed
  significant_genes <- significant_genes[significant_genes %in% cmplot_data$SNP]
  
  # Generate Manhattan plot
  CMplot(
    cmplot_data,
    plot.type = "m",                # Manhattan plot
    LOG10 = TRUE,                   # -log10(P-value)
    threshold = threshold, # Significance threshold
    threshold.col = "red",           # Threshold line color
    threshold.lty = 1,               # Threshold line style (dashed)
    chr.den.col = NULL,              # No background density
    amplify = TRUE,                  # Enlarge significant points
    highlight = significant_genes,   # Highlight significant genes
    highlight.col = "blue",           # Highlight color
    highlight.cex = 1.5,             # Highlight point size
    highlight.text = significant_genes,            # Add gene labels
    highlight.text.cex = 0.8,         # Text size for labels
    signal.cex = 1.5,
    file.output = TRUE,              # Save plot to file
    file = "jpg",                    # Output format ("pdf" or "jpg")
    dpi = 300, 
    file.name = output_name, # High resolution
    #outdir = "./",                   # Output directory (current folder)
    verbose = TRUE                   # Print progress
  )
  
  message("Manhattan plot saved as: ", output_name, ".jpg")
}
view_4ch_magma<-fread('./4ch_3Duin/magma.genes.out',data.table = FALSE)
view_2ch_magma<-fread('./2ch_3Duin/magma.genes.out',data.table = FALSE)
plot_magma_manhattan_with_labels('./4ch_3Duin/magma.genes.out',output_name = "4ch_3Duin")
plot_magma_manhattan_with_labels('./2ch_3Duin/magma.genes.out',output_name = "2ch_3Duin")
view_4ch_magma<-view_4ch_magma[which(view_4ch_magma$P<(0.05/19291)),]
## 2ch view function ananlysis
view_2ch_magma<-view_2ch_magma[which(view_2ch_magma$P<(0.05/19291)),]
intersect(view_4ch_magma$SYMBOL,heart_gene$symbol)%>%length()
intersect(view_2ch_magma$SYMBOL,heart_gene1$symbol)%>%length()
##
write.csv(view_4ch_magma,'magma_gene_4ch.csv',row.names = FALSE)
write.csv(view_2ch_magma,'magma_gene_2ch.csv',row.names = FALSE)


##
###
venn_list<-list(MAGMA_4Ch=view_4ch_magma$SYMBOL,MAGMA_2Ch=view_2ch_magma$SYMBOL,eQTL_2ch=heart_gene1$symbol,eQTL_4ch=heart_gene$symbol)
save_name<-'heart_gene_overlap'
save_name<-paste(save_name,'.png',sep = "")
venn.plot <- venn.diagram(
  x = venn_list,
  category.names = c("MAGMA_4Ch", "MAGMA_2Ch", "eQTL_2Ch", "eQTL_4Ch"),
  filename = save_name, 
  output = TRUE,
  imagetype = "png",
  height = 3000,
  width = 3500,
  resolution = 500,
  fill = c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"),  # 4 colors for 4 sets
  alpha = 0.6,
  cex = 1.5,            # Font size inside circles
  cat.cex = 1,        # Font size for category names
  cat.col = c("black", "black", "black", "black"),  # 4 colors for 4 labels
  lwd = 1.5,
  lty = "solid"
)


view_2ch_function<-plot_function_enrichment('./2ch_3D_GAN/magma.gsa.out',output_file ='2ch_functional_enrichment.pdf')
plot_disease_enrichment('./heart_function.txt',output_file ='4ch_phenotype_enrichment.pdf' )
plot_tissue_enrichment('./2ch_3D_GAN/magma_exp_gtex_v8_ts_avg_log2TPM.gsa.out',output_file ='2ch_tissue_enrichment.pdf' )
plot_celltype_enrichment('./2ch_3D_GAN_heart_celltype/magma_celltype_step1.txt',output_file ='2ch_cell_type_enrichment.pdf' )
###
view_2Ch_gene<-fread('./2ch_3Duin/genes.txt')
view_4Ch_view_gene<-fread('./4ch_3Duin/genes.txt')
###
view_2Ch_gene_eqtl<-view_2Ch_gene[(!is.na(view_2Ch_gene$eqtlMapminP))|(view_2Ch_gene$ciMap=='Yes'),]
view_4Ch_gene_eqtl<-view_4Ch_view_gene[(!is.na(view_4Ch_view_gene$eqtlMapminP))|(view_4Ch_view_gene$ciMap=='Yes'),]
#
view_2Ch_eqtl_snp<-view_2Ch_gene_eqtl$IndSigSNPs%>%as.character()
view_4Ch_eqtl_snp<-view_4Ch_gene_eqtl$IndSigSNPs%>%as.character()
# clearn function
extract_rsids <- function(rs_column) {
  rs_column %>%
    gsub(" ", "", .) %>%               
    gsub(":", ";", .) %>%                
    paste(collapse = ";") %>%               
    strsplit(";") %>%                        
    unlist() %>%                            
    unique() %>%                             
    sort()                                   
}
# perform the filter
view_2Ch_eqtl_snp <- extract_rsids(view_2Ch_eqtl_snp)
view_4Ch_eqtl_snp<-extract_rsids(view_4Ch_eqtl_snp)
##
intersect(res_4ch_gan$rsID,view_4Ch_eqtl_snp)%>%length()
intersect(res_2ch_gan$rsID,view_2Ch_eqtl_snp)%>%length()
##
###
### PRS plot  ##

PRS2_2Ch_files<-fread('../2Ch_PRS.csv',sep = ',',data.table = FALSE)
PRS_4Ch_files<-fread('../4Ch_PRS.csv',sep = ',',data.table = FALSE)
PRS2_2Ch_files$FDR<--log10(p.adjust(PRS2_2Ch_files$p_value,method = 'bonferroni'))
PRS_4Ch_files$FDR<--log10(p.adjust(PRS_4Ch_files$p_value,method = 'bonferroni'))
PRS2_2Ch_files<-PRS2_2Ch_files[order(PRS2_2Ch_files$FDR),]
PRS_4Ch_files<-PRS_4Ch_files[order(PRS_4Ch_files$FDR),]

##
PRS2_2Ch_files<-PRS2_2Ch_files[-6,]
PRS_4Ch_files<-PRS_4Ch_files[-6,]
type_all<-c(rep('2Ch',6),rep('4Ch',6))
PRS_plot_df<-rbind(PRS2_2Ch_files,PRS_4Ch_files)
PRS_plot_df$group<-type_all
# Plot using ggscatter from ggpubr package
ggscatter(PRS_plot_df, 
          x = "prs_file",                  # x-axis: PRS file names
          y = "FDR",     # y-axis: canonical correlation
          size = 6,  # Bubble size based on canonical correlation
          facet.by='group',
          color = "canonical_correlation",               # Bubble color based on p-value
          gradient.color = c("red", "blue"),                # Color palette (Red to Blue)
          alpha = 0.8,                     # Transparency of bubbles
     #     label = "prs_file",              # Label points with PRS file names
          repel = TRUE) +                  # Avoid overlapping labels
  labs(title = "Canonical Correlation between disorder PRS and UDIP-heart",
       x = "PRS File",
       y = "-log10(Adjusted P)",
       color = "Canonical correlation",
       size = "-log10(Adjusted P)") +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))+coord_flip()+ geom_hline(yintercept = 1.3, color = "red", linetype = "dashed", size = 1)
##

### heart related gene analysis 
heart_disorder_gene<-fread('../Cardiac_Disease_Genes__Excluding_Traits_.csv',data.table = FALSE)
diorder_gene_sta<-table(heart_disorder_gene$`Disease/Trait ID`)%>%as.data.frame()
diorder_gene_choose<-diorder_gene_sta[which(diorder_gene_sta$Freq>200),]
###  ###
diorder_gene_selected<-heart_disorder_gene[which(heart_disorder_gene$`Disease/Trait ID`%in%diorder_gene_choose$Var1),]
### background gene ##
background_gene<-fread('../NCBI37.3.gene.loc',data.table = FALSE)
##
# venn_list<-list(MAGMA_4Ch=view_4ch_magma$SYMBOL,MAGMA_2Ch=view_2ch_magma$SYMBOL,eQTL_2ch=heart_gene1$symbol,eQTL_4ch=heart_gene$symbol)
all_disorder_name<-diorder_gene_selected$`Disease/Trait`%>%unique()
gene_4Ch<-union(view_4ch_magma$SYMBOL,heart_gene$symbol) ##  652
gene_2Ch<-union(view_2ch_magma$SYMBOL,heart_gene1$symbol)## 628
##
union(gene_4Ch,gene_2Ch)
##
select_disease<-c('Dilated cardiomyopathy','Hypertrophic cardiomyopathy')
for(i in all_disorder_name){
  #i=all_disorder_name[1]
  heart_dis_temp<-diorder_gene_selected[which(diorder_gene_selected$`Disease/Trait`==i),]
  fisherVennPlot(heart_dis_temp$Gene,gene_2Ch,background_gene$V6,name1=i,name2='2Ch Heart-UDIP',paste('./disoder_gene_overlap/2Ch/',i,'_2Ch_UDIP.png',sep = ""))
}

### single cell TRN ##
singlecell_heart_trn<-fread('../single_cell_heart_TRN.csv')
# explore the association with HCM in single cell heart TRN ,focus on the UDIP_heart regulate the disease related gene

UDIP_tf<-intersect(singlecell_heart_trn$`TF name`,gene_4Ch)
other_tf<-setdiff(singlecell_heart_trn$`TF name`,gene_4Ch)
UDIP_tf_subnet<-singlecell_heart_trn[(singlecell_heart_trn$`TF name`%in%gene_4Ch)|(singlecell_heart_trn$`gene name`%in%gene_4Ch),]
HCM_gene<-diorder_gene_selected[which(diorder_gene_selected$`Disease/Trait`=='Hypertrophic cardiomyopathy'),2]
###
gene_dis<-union(HCM_gene,gene_4Ch)
UDIP_tf_subnet_HCM<-singlecell_heart_trn[which((singlecell_heart_trn$`gene name`%in%gene_dis)& (singlecell_heart_trn$`TF name`%in%gene_dis)),]
other_rd<-singlecell_heart_trn[singlecell_heart_trn$`TF name`%in%other_tf,]

####

###
pos_4Ch<-heart_gene[which(heart_gene$posMapSNPs!=0),]
pos_2Ch<-heart_gene1[which(heart_gene1$posMapSNPs!=0),]
genomic_loc<-fread('2Ch_3D_GAN/GenomicRiskLoci.txt')
intersect(pos_4Ch$IndSigSNPs,res_4ch_gan$rsID)
####compare result #
Aung_res<-fread('../comparsion/Aung.csv',data.table = FALSE)
Bonazzola_res<-fread('../comparsion/Aung.csv',data.table = FALSE)
meyer_res<-fread('../comparsion/Meyer.csv',data.table = FALSE)
#
meyer_res<-meyer_res[which(meyer_res$Locus!='-'),]
Burns_res<-fread('../comparsion/PCA_left_MAGMA_gene.csv',data.table = FALSE)
Burns_res_lead_snp<-fread('../comparsion/PCA_left_lead_SNP.csv',data.table = FALSE)
PCA_fuma<-fread('../comparsion/PCA_FUMA.csv',data.table = FALSE)
Burns_res<-union(Burns_res$`Gene symbol`,PCA_fuma$Locus)
Burns_res<-Burns_res[which(Burns_res!="")]
Pirru_res<-fread('../comparsion/Pirruccello.csv',data.table = FALSE)
Pirru_res<-union(Pirru_res$Gene1,Pirru_res$Gene2)
Pirru_res<-Pirru_res[which(Pirru_res!="")]
####
Sooknah_log<-fread('../comparsion/Sooknah_et_al_log.csv')
Sooknah_short<-fread('../comparsion/Sooknah_short.csv')
##
Sooknah_lead_snp<-union(Sooknah_log$`Lead Variant`,Sooknah_short$`Lead Variant`)%>%unique()
Sooknah_gene<-union(Sooknah_log$`Gene Name`,Sooknah_short$`Gene Name`)%>%unique()


library(UpSetR)

plot_upset <- function(set_list, set_names = NULL, nsets = 5, nintersects = 20, 
                       order.by = "freq", mainbar.y.label = "Intersection Size",
                       sets.x.label = "Set Size", keep.order = FALSE) {
  # set_list: A named list where each element is a vector representing a set
  # set_names: Optional. A vector of names for the sets if set_list has no names
  # nsets: Number of individual sets to show on the x-axis
  # nintersects: Number of intersections to show in the main bar plot
  # order.by: How to order the intersections ("freq" or "degree")
  # mainbar.y.label: Label for the y-axis of the intersection size barplot
  # sets.x.label: Label for the x-axis (set sizes)
  # keep.order: Whether to keep the original input order of sets
  
  # Check if set names are provided
  if (is.null(names(set_list))) {
    if (is.null(set_names)) stop("Please provide either named set_list or set_names.")
    names(set_list) <- set_names
  }
  
  # Get all unique elements across all sets
  all_elements <- unique(unlist(set_list))
  
  # Create a binary presence/absence dataframe for UpSetR
  upset_df <- data.frame(element = all_elements)
  for (set_name in names(set_list)) {
    upset_df[[set_name]] <- as.integer(all_elements %in% set_list[[set_name]])
  }
  
  # Plot the UpSet diagram
  UpSetR::upset(upset_df, 
        nsets = nsets, 
        nintersects = nintersects, 
        order.by = order.by,
        mainbar.y.label = mainbar.y.label,
        sets.x.label = sets.x.label,
        keep.order = keep.order)
}
###
test_dd<-apply(upset_df[2:ncol(upset_df)], 1, sum)
###
names(test_dd)<-upset_df[,1]
test_dd<-as.data.frame(test_dd)
test_dd$gene<-rownames(test_dd)
##
test_dd<-test_dd[which(test_dd$test_dd==1),]
##
gene_one<-test_dd$gene
cc<-upset_df[which((upset_df$UIDP_2Ch==1)|(upset_df$UIDP_4Ch==1)),]
intersect(cc$element,gene_one)

###
plot_upset_advanced <- function(set_list, set_colors = NULL, top_n = 5, mainbar.y.label = "Intersection Size") {
  # set_list: named list where each element is a vector (one set)
  # set_colors: optional vector of colors (one color per set)
  # top_n: number of top intersection elements to label
  # mainbar.y.label: y-axis label for intersection size bar plot
  
  if (is.null(names(set_list))) {
    stop("set_list must be a named list with names for each set.")
  }
  
  # Prepare the data
  all_elements <- unique(unlist(set_list))
  upset_data <- data.frame(element = all_elements)
  
  for (set_name in names(set_list)) {
    upset_data[[set_name]] <- as.integer(all_elements %in% set_list[[set_name]])
  }
  
  # Find the intersection size
  upset_data_long <- upset_data %>%
    pivot_longer(cols = -element, names_to = "Set", values_to = "Present") %>%
    filter(Present == 1)
  
  intersection_counts <- upset_data_long %>%
    group_by(element) %>%
    summarise(sets_in = n()) %>%
    arrange(desc(sets_in))
  
  # Get top_n elements appearing in the most sets
  top_elements <- intersection_counts$element[1:min(top_n, nrow(intersection_counts))]
  
  # Check colors
  if (!is.null(set_colors)) {
    if (length(set_colors) != length(set_list)) {
      stop("Length of set_colors must match number of sets.")
    }
    names(set_colors) <- names(set_list)
  }
  
  # Create UpSet plot
  p <- upset(
    upset_data,
    intersect = names(set_list),
    name = "Genes",
    base_annotations = list(
      'Intersection size' = intersection_size(
        counts = TRUE
      ) +
        ylab(mainbar.y.label)
    ),
    set_sizes = (
      upset_set_size(aes(fill = after_stat(set))) +  # ⭐️ important: fill set size by set
        theme(axis.text.x = element_text(angle = 90))
    )
  )
  
  # Apply color if provided
  if (!is.null(set_colors)) {
    p <- p + scale_fill_manual(values = set_colors)
  }
  
  # Print plot
  print(p)
  
  # Print top elements
  cat("Top elements appearing in most sets:\n")
  print(top_elements)
}

nature_colors <- c(
  "#3C5488", # Blue
  "#E64B35", # Orange
  "#00A087", # Green
  "#B71C1C", # Red
  "#7E57C2", # Purple
  "#FFC107", # Yellow
  "#4DBBD5"  # Cyan
)


# Draw the UpSet plot
plot_upset(
  set_list = list(UIDP_2Ch = gene_2Ch, UIDP_4Ch = gene_4Ch,Sooknah_et_al=Sooknah_log$`Gene Name`,Aung_et_al = Aung_res$`Locus Name`, Bonazzola_et_al= Bonazzola_res$`Locus Name`,
                  Meyer_et_al=meyer_res$Locus,Burns_et_al=Burns_res, Pirru_et_al=Pirru_res),
  nsets = 8,
  nintersects = 200,
  order.by = "freq"
)
## lead SNP
plot_upset(
  set_list = list(UIDP_2Ch = res_2ch_gan$rsID, UIDP_4Ch = res_4ch_gan$rsID, Sooknah_et_al=Sooknah_log$`Lead Variant`,Aung_et_al = Aung_res$`Lead variant`,Bonazzola_et_al= Bonazzola_res$`Lead variant`,
                  Meyer_et_al=meyer_res$SNP, Burns_et_al=Burns_res_lead_snp$`Lead SNV`,Pirru_et_al=Pirru_res$`Lead SNP`),
  nsets = 8,
  nintersects = 200,
  order.by = "freq"
)
###
write.table(heart_gene$symbol,'4Ch_eqtl_gene.txt',row.names =FALSE,col.names = FALSE,quote = FALSE)
write.table(heart_gene1$symbol,'2Ch_eqtl_gene.txt',row.names =FALSE,col.names = FALSE,quote = FALSE)
### associaiton withn traditional feature ###
####
Association_traditon<-fread('Associaiton_tradition_res.csv')
dd<-str_split(Association_traditon$PRS_column,'p')%>%unlist()
##

library(locuscomparer)
library(ggrepel)
##

all_4ch_view<-fread('HEART_4ich_final_inputation_3Duin.txt',data.table = FALSE)
HCM_all<-fread('HCM.tsv',data.table = FALSE)
HCM_all <- subset(HCM_all, chromosome == 6 & base_pair_location >= 36444305 & base_pair_location <= 36855116)
all_4ch_view <- subset(all_4ch_view, CHR == 6 & POS >= 36444305 & POS <= 36855116)
HCM_all<-HCM_all[,c(9,8)]
all_4ch_view<-all_4ch_view[,c(2,8)]
colnames(HCM_all)<-c('rsid','pval')
colnames(all_4ch_view)<-c('rsid','pval')
write.table(all_4ch_view,'all_4ch_view_CDKN1A.txt',row.names = FALSE,quote = FALSE,sep = '\t')
write.table(HCM_all,'HCM_plot.tsv_CDKN1A',row.names = FALSE,quote = FALSE,sep = '\t')
locuscompare(in_fn1 = "./all_4ch_view_CDKN1A.txt",  # 
             in_fn2 = "./HCM_plot.tsv_CDKN1A", 
             title1 = "4CH view",
             title2 = "HCM",
             genome='hg19',
            # snp = "rs17617337"
             # region = "chr7:120750000-121200000",
             # region = "chr9:120750000-121200000", 
            )  # 主SNP
##
intersect(HCM_all$rsid,'rs17617337')
snp = "rs17617337"
## genomic regions ###
genomic_4Ch<-fread('./4ch_3Duin/gwascatalog.txt')
genomic_2Ch<-fread('./2ch_3Duin/gwascatalog.txt')
###
write.csv(genomic_4Ch,'gwas_categlog_4Ch.csv',row.names = FALSE)
write.csv(genomic_2Ch,'gwas_categlog_2Ch.csv',row.names = FALSE)
## all gene  #####
Gene_4Ch<-fread('./4ch_3Duin/genes.txt')
Gene_2Ch<-fread('./2ch_3Duin/genes.txt')
###
write.csv(Gene_4Ch,'gene_4Ch.csv',row.names = FALSE)
write.csv(Gene_2Ch,'gene_2Ch.csv',row.names = FALSE)














