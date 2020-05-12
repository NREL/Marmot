#Read in file locations
library(data.table)
inputs = data.table(read.csv('Marmot_user_defined_inputs.csv', header = T))
parent_dir = as.character(inputs[Input == 'Processed_Solutions_folder']$User_defined_value)
scenario_dir = as.character(inputs[Input == 'Main_scenario_plot']$User_defined_value)
agg_by_prefix = as.character(inputs[Input == 'AGG_BY']$User_defined_value)

output.dir = paste0(parent_dir,'/',scenario_dir)

rmarkdown::render(input = file.path('HTML_out.Rmd'),c('html_document'),
                  output_file = paste0(scenario_dir,'_by ',agg_by_prefix,'.html'),output_dir = output.dir)
