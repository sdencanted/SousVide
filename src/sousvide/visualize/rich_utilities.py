import numpy as np

from rich import get_console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from typing import Tuple,List

console = get_console()

def get_generation_progress() -> Progress:
    """
    Create a Rich Progress instance for visualizing generation tasks.
    
    Returns:
        progress:   A Progress object configured for generation tasks.
    """
    
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[bold green3] {task.completed:>2}/{task.total} {task.fields[units]}"),
        TimeRemainingColumn(elapsed_when_finished=True),
        console=console,
        auto_refresh=False,
    )
    return progress

def get_training_progress() -> Progress:
    """
    Create a Rich Progress instance for visualizing training tasks.
    
    Returns:
        progress:   A Progress object configured for training tasks.
    """
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description} | [bold dark_green]Loss[/]: [dark_green]{task.fields[loss]:.4f}[/]"),
        TextColumn(
            "[bold bright_green]TTE[/]: "
            "[bright_green]{task.fields[tte]}[/]"),
        BarColumn(),
        TextColumn("[bold green3] {task.completed:>2}/{task.total} {task.fields[units]}"),
        TextColumn("[bold cyan]ETA[/]: [cyan]{task.fields[eta]}[/]"),
        TextColumn("[cyan]({task.fields[remaining]} remaining)[/]"),
        console=console,
        auto_refresh=False,
    )
    return progress

def get_data_description(name:str,value:int, subunits:str=None, Nmn:int=10,Nct:int=10) -> str:
    """"
    Generate a description for the data progress.
    
    Args:
        name:   Name of the data.
        value:  Number of data points."

    Returns:
        data_desc:   Formatted string describing the course and number of data points.
    """


    name_desc = f"[bold green3]{name:<{Nmn}}[/][dark_green]"

    if subunits is not None:
        value_desc = f"({value:>4} {subunits})".ljust(Nct)
    else:
        value_desc = f"{value:>4.5}"

    data_desc = name_desc + value_desc
    
    return data_desc

def get_deployment_table() -> str:
    """
    Generate a table header for deployment data.
    
    Returns:
        table:   Formatted string for the deployment table header.
    """

    # Create a table
    table = Table(title=f"Deployment Summary")

    # Add columns
    table.add_column("Pilot", justify="left")
    table.add_column("TTE Mean", justify="center")
    table.add_column("TTE Best", justify="center")
    table.add_column("PP", justify="center")
    table.add_column("Hz Mean", justify="center")
    table.add_column("Hz Worst", justify="center")

    return table

def update_deployment_table(table:Table,pilot:str,metrics:dict):
    """
    Update the deployment table with metrics data.
    
    Args:
        table:   The deployment table to update.
        pilot:   Name of the pilot.
        metrics:   Dictionary containing the metrics data.
        
    Returns:
        table:   Updated deployment table with metrics.
    """
    
    # Add a row with the metrics
    table.add_row(
        pilot,
        f"{metrics['TTE']['mean']:.2f}",
        f"{metrics['TTE']['best']:.2f}",
        f"{metrics['PP']:.2f}",
        f"{metrics['hz']['mean']:.2f}",
        f"{metrics['hz']['worse']:.2f}"
    )
    
    return table

def get_student_summary(student:str,
                        Neps_tot:int,Nd_mean:Tuple[int],T_tn_tot:int,
                        loss_tn:float,loss_tt:float,eval_tte:float|None,
                        Nln:int=70) -> str:
    
    # Compute the training time
    hours = T_tn_tot // 3600
    minutes = (T_tn_tot % 3600) // 60
    seconds = T_tn_tot % 60

    # Summary First Line
    student_field = f"Student: [bold cyan]{student}[/]"
    tepochs_field = f"Epochs: {Neps_tot}"
    datsize_field = f"Data Size: {Nd_mean[0]}/{Nd_mean[1]}"

    summary = (
            f"{'-' * Nln}\n"
            f"{student_field:<33} | {tepochs_field:<13} | {datsize_field:<40}\n"
    )

    # Summary Second Line
    t_field = f"Time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
    tn_field = f"[bold bright_green]Train: {loss_tn:.4f}[/]"
    tt_field = f"Test: {loss_tt:.4f}"

    if eval_tte is not None:
        ev_field = f"[bold bright_green]Eval TTE: {eval_tte:.2f}[/]"
        eval_field = (f"{t_field:<19} | {tn_field:<35} | {tt_field:<10} | {ev_field}\n")
    else:
        eval_field = (f"{t_field:<19} | {tn_field:<35} | {tt_field:<10}\n")

    summary += eval_field            
    
    return summary
