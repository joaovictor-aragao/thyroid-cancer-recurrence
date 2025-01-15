import plotly.graph_objects as go

PLOT_COLORS = ["#3288BD", "#99D594", "#E6F598", "#FEE08B", "#FC8D59", "#D53E4F"]

def create_barplots(dataframe, varname):

    l1 = dataframe[dataframe["Recurred"]== 'No'][varname].value_counts().sort_index().index.values
    l2 = dataframe[dataframe["Recurred"]== 'No'][varname].value_counts().sort_index().values

    l3 = dataframe[dataframe["Recurred"]== 'Yes'][varname].value_counts().sort_index().index.values
    l4 = dataframe[dataframe["Recurred"]== 'Yes'][varname].value_counts().sort_index().values

    trace0 = go.Bar(
        x = l1,
        y = l2,
        name='No recurred',
        marker_color=PLOT_COLORS[0]
    )

    # Second plot
    trace1 = go.Bar(
        x = l3,
        y = l4,
        name="Recurred",
        marker_color=PLOT_COLORS[1]
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title=varname
    )

    return go.Figure(data=data, layout=layout)