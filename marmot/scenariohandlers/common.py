


def get_winter_summer_peaks(dispatch, winter_months=[1,2,12]):


    winter_mask = dispatch.index.month.isin(set(winter_months))

    vre_cols = [tech for tech in qp.curt_tech if tech in dispatch.columns.get_level_values('Technology').unique()]
    total_demand = dispatch.groupby(axis=1, level='Technology').sum()['Demand']
    total_vre = dispatch.groupby(axis=1, level='Technology').sum()[vre_cols].sum(axis=1)
    total_net_load = total_demand - total_vre

    peak_summer_demand = total_demand[~winter_mask].idxmax()
    peak_summer_netload = total_net_load[~winter_mask].idxmax()

    peak_winter_demand = total_demand[winter_mask].idxmax()
    peak_winter_netload = total_net_load[winter_mask].idxmax()

    peak_days = {
        'Winter Peak Demand': peak_winter_demand,
        'Winter Peak Net Load': peak_winter_netload,
        'Summer Peak Demand': peak_summer_demand,
        'Summer Peak Net Load': peak_summer_netload
        }

    return peak_days