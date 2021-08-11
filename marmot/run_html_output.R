#Read in file locations
start_time = Sys.time()
library(data.table)
inputs = fread('Marmot_user_defined_inputs.csv', header = T)
parent_dir = as.character(inputs[Input == 'Marmot_Solutions_folder']$User_defined_value)
agg_level = as.character(inputs[Input == 'AGG_BY']$User_defined_value)
title = tail(strsplit(parent_dir,'/')[[1]], n = 1)
rmarkdown::render(input = file.path('HTML_out.Rmd'),c('html_document'),
                  output_file = paste0(parent_dir,'.html'),output_dir = parent_dir,
                  output_options = list(pandoc_args = c(paste0('--metadata=title:',title))))
message(paste0('HTML compiled in ',Sys.time() - start_time,'minutes.'))
