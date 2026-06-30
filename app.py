from shiny import App, ui, reactive, render
import requests
import pandas as pd
import plotly.io as pio
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:5000/ask-card"

def empty_figure(message="No figure"):
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )

    fig.update_layout(
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig

def to_plotly_figure(fig_obj):
    if fig_obj is None:
        return empty_figure()

    if isinstance(fig_obj, dict):
        plotly_dict = fig_obj.get("plotly_json", fig_obj)
        return go.Figure(plotly_dict)

    return empty_figure()


app_ui = ui.page_fluid(

    ui.tags.style("""
    .panel-container {
        display: flex;
        gap: 12px;
        align-items: stretch;
        width: 100%;
    }

    .resizable-panel {
        resize: horizontal;
        overflow: auto;
        min-width: 300px;
        width: 33%;
        flex: 0 0 auto;
    }
    """),

    ui.h2("AI Analytics Engine"),

    ui.input_text(
        "query",
        "Question",
        placeholder="Describe steam cost for grade 6010120 in week 18"
    ),

    ui.input_action_button(
        "ask",
        "Ask"
    ),

    ui.hr(),

    ui.div(
        ui.card(
            ui.card_header("Markdown"),
            ui.output_ui("markdown"),
            class_="resizable-panel",
        ),
        ui.card(
            ui.card_header("Tables"),
            ui.output_ui("tables"),
            class_="resizable-panel",
        ),
        ui.card(
            ui.card_header("Figures"),
            ui.navset_tab(
                ui.nav_panel("Figure 1", output_widget("fig_0")),
                ui.nav_panel("Figure 2", output_widget("fig_1")),
                ui.nav_panel("Figure 3", output_widget("fig_2")),
                ui.nav_panel("Figure 4", output_widget("fig_3")),
                ui.nav_panel("Figure 5", output_widget("fig_4")),
                ui.nav_panel("Figure 6", output_widget("fig_5")),
                ui.nav_panel("Figure 7", output_widget("fig_6")),
            ),
            class_="resizable-panel",
        ),
        class_="panel-container",
    )
)

def server(input, output, session):

    result = reactive.Value(None)

    
    def get_figure(i):
        res = result.get()
        if not res:
            return empty_figure()

        figs = res.get("figures") or []

        if i >= len(figs):
            return empty_figure()

        return to_plotly_figure(figs[i])

    @reactive.effect
    @reactive.event(input.ask)
    def _():

        r = requests.post(
            API_URL,
            json={
                "query": input.query(),
                "download_artifacts": False,
                "diagnosis_summary": True,
                "cost_driver_summary": False,
            }
        )

        r.raise_for_status()

        result.set(r.json())

    @output
    @render.ui
    def markdown():

        res = result.get()

        if res is None:
            return ui.p("No response yet.")

        return ui.markdown(res.get("text", ""))

    @output
    @render.ui
    def tables():

        res = result.get()

        if res is None:
            return ui.p()

        tables = res.get("tables", [])

        if not tables:
            return ui.p("No tables")

        tabs = []

        for i, t in enumerate(tables):
            if "inline" in t and t["inline"]:
                df = pd.DataFrame(t["inline"]["data"])
                tabs.append(
                    ui.nav_panel(
                        f"Table {i+1}",
                        ui.HTML(
                            df.head(100).to_html(
                                classes="table table-striped"
                            )
                        )
                    )
                )
            else:
                continue


        return ui.navset_tab(*tabs)
    
    @output
    @render_widget
    def fig_0():
        return get_figure(0)

    @output
    @render_widget
    def fig_1():
        return get_figure(1)

    @output
    @render_widget
    def fig_2():
        return get_figure(2)

    @output
    @render_widget
    def fig_3():
        return get_figure(3)

    @output
    @render_widget
    def fig_4():
        return get_figure(4)

    @output
    @render_widget
    def fig_5():
        return get_figure(5)


app = App(app_ui, server)