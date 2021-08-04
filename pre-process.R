library(Seurat)
library(R.utils)
# download the 199 mouse testis spermatocyte cells from GSE113293 (PMID:31237565)
# file name: GSM3102983_SPCImat.txt

example_data<- read.table(file = 'GSM3102983_SPCImat.txt',header = T,stringsAsFactors = F)
cellname<- paste(rep('C',times = ncol(example_data)),1:ncol(example_data),sep = '_')
colnames(example_data)<- cellname
# load NCBI gene information
geneinfo<- readRDS('R/geneinfo.rds')
geneinfo<- geneinfo[geneinfo$species == 'Mouse',]
# revise gene symbol
genename<- rownames(example_data)
genename1<- genename[genename %in% geneinfo$Symbol]
genename2<- genename[!genename %in% geneinfo$Symbol]
genename3<- genename2[genename2 %in% geneinfo$Synonyms]
genename4<- rep('NA',length(genename3))

for (i in 1:length(genename3)) {
  d1<- geneinfo[geneinfo$Synonyms == genename3[i],]$Symbol
  if(length(d1) == 1){
    genename4[i]<- d1
  }
}
genename3<- c(genename1,genename3)
genename4<- c(genename1,genename4)
genedata<- data.frame(raw_name = genename3,new_name = genename4,stringsAsFactors = F)
genedata<- genedata[!genedata$new_name == 'NA',]
genedata1<- as.data.frame(table(genedata$new_name),stringsAsFactors = F)
genedata1<- genedata1[genedata1$Freq == 1,]
genedata<- genedata[genedata$new_name %in% genedata1$Var1,]
example_data <- CreateSeuratObject(counts = example_data)
example_data <- NormalizeData(object = example_data)

# generate example_data.rds
saveRDS(example_data,file = 'R/example_data.rds')

# example_data is Seurat object after log-normalization
mouse_Testis199<- readRDS('R/example_data.rds')
mouse_Testis199<- mouse_Testis199[['RNA']]@data

# revising gene symbols
genename<- rownames(mouse_Testis199)
genename1<- genename[genename %in% geneinfo$Symbol]
genename2<- genename[!genename %in% geneinfo$Symbol]
genename3<- genename2[genename2 %in% geneinfo$Synonyms]
genename4<- rep('NA',length(genename3))
for (i in 1:length(genename3)) {
  d1<- geneinfo[geneinfo$Synonyms == genename3[i],]$Symbol
  if(length(d1) == 1){
    genename4[i]<- d1
  }
}
genename3<- c(genename1,genename3)
genename4<- c(genename1,genename4)
genedata<- data.frame(raw_name = genename3,new_name = genename4,stringsAsFactors = F)
genedata<- genedata[!genedata$new_name == 'NA',]
genedata1<- as.data.frame(table(genedata$new_name),stringsAsFactors = F)
genedata1<- genedata1[genedata1$Freq == 1,]
genedata<- genedata[genedata$new_name %in% genedata1$Var1,]
mouse_Testis199<- mouse_Testis199[genedata$raw_name,]
all(rownames(mouse_Testis199) == genedata$raw_name)
rownames(mouse_Testis199)<- genedata$new_name
all(rownames(mouse_Testis199) == genedata$new_name)
all(rownames(mouse_Testis199) %in% geneinfo$Symbol)

# for log-normalization, please execute the following command
# mouse_Testis199<- CreateSeuratObject(counts = mouse_Testis199)
# mouse_Testis199<- NormalizeData(object = mouse_Testis199)
# mouse_Testis199<- mouse_Testis199[['RNA']]@data

mouse_Testis199<- as.matrix(mouse_Testis199)
write.csv(mouse_Testis199,file = 'test/mouse/mouse_Testis199_data.csv')
gzip('test/mouse/mouse_Testis199_data.csv','test/mouse/mouse_Testis199_data.gz')

# mouse_Testis199_data.csv or mouse_Testis199_data.gz can be used for running scDeepSort
