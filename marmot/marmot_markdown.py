import markdown as md
from markdown.extensions.toc import TocExtension
import os
import pandas as pd
import sys
# try:
#     import marmot.utils.mconfig as mconfig
# except ModuleNotFoundError:
#     from utils.definitions import INCORRECT_ENTRY_POINT

#     print(INCORRECT_ENTRY_POINT.format(Path(__file__).name))
#     sys.exit()
# from marmot.utils.definitions import INPUT_DIR

# Marmot_user_defined_inputs = pd.read_csv(
#     INPUT_DIR.joinpath(mconfig.parser("user_defined_inputs_file")),
#     usecols=["Input", "User_defined_value"],
#     index_col="Input",
#     skipinitialspace=True,
# )

# figures_dir = os.path.join(
#     (
#     Marmot_user_defined_inputs.loc["Model_Solutions_folder"]
#     .to_string(index=False)
#     .strip()
#     ),
#     'Figures_output')


markdown_title = 'CSU_FLECCS_PLEXOS_results'
figures_dir = '/Volumes/ReEDS/FY22-ARPAE_FLECCS/CSU_202202/Runs_ColdStorageFix/R2P_solutions/Figures_Output'

static_cat_order = ['total_installed_capacity',
                    'total_generation',
                    'generation_stack',
                    'production_cost_full',
                    'production_cost',
                    'transmission',
                    'ramping',
                    'curtailment',
                    'emissions']

static_agg_order = ['Summary',
                    'Country',
                    'zone',
                    'region']

avail_cats = []
for cat in os.listdir(figures_dir):
    if os.path.isdir(os.path.join(figures_dir,cat)):
        if cat.count('_') > 1:
            cat_new = '_'.join([el for el in cat.split('_')[1:]])
        else:
            cat_new = cat.split('_')[1]
        if cat_new not in avail_cats:
            avail_cats.append(cat_new)

#Sort lists
avail_cats.sort(key = lambda i: static_cat_order.index(i))

##Initialize markdown file
f = open(os.path.join(figures_dir,f'{markdown_title}.md'), 'bw+')
f.write(f'# {markdown_title}\n'.encode('utf-8'))
f.write('[TOC]\n'.encode('utf-8'))

##Loop through plotting categories.
for cat in avail_cats:
    f.write(f'## {cat}\n'.encode('utf-8'))
    aggs = [c.split('_')[0] for c in os.listdir(figures_dir) if cat in c] #Get aggregations used for this plotting category.
    aggs.sort(key = lambda i: static_agg_order.index(i))
    for agg in aggs:
        f.write(f'### {agg}\n'.encode('utf-8'))
        agg_dir = os.path.join(figures_dir,f'{agg}_{cat}')
        output_files = os.listdir(agg_dir)
        for file in output_files:
            if file.endswith('.svg') or file.endswith('.png'):
                f.write(f'![]({agg_dir}/{file})\n'.encode('utf-8'))

f.seek(0)
md.markdownFromFile(input=f, 
                    output=os.path.join(figures_dir,f'{markdown_title}.html'),
                    extensions = [TocExtension(baselevel=3)])