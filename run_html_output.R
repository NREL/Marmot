#Read in file locations
start_time = Sys.time()
#setwd("C:/Users/mschwarz/Desktop/Marmot")
setwd('/home/mschwarz/PLEXOS results analysis/Marmot')
library(data.table)
inputs = data.table(read.csv('Marmot_user_defined_inputs.csv', header = T))
parent_dir = as.character(inputs[Input == 'Processed_Solutions_folder']$User_defined_value)
rmarkdown::render(input = file.path('HTML_out.Rmd'),c('html_document'),
                  output_file = paste0(parent_dir,'test.html'),output_dir = parent_dir)
message(paste0('HTML compiled in ',Sys.time() - start_time,'minutes.'))
