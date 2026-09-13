"""Nova CLI entry point."""

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from .config import AnimationConfig
from .config.loader import create_animation_config, load_config
from .config.schemas import ModelConfig

app = typer.Typer(
    name="nova",
    help="Nova: AI-Assisted 2D Anime Animation System",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool):
    if value:
        from . import __version__
        console.print(f"Nova v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", callback=version_callback, is_eager=True, help="Show version"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """Nova: Progressive frame-by-frame 2D anime animation."""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


@app.command()
def config(
    start_image: Path = typer.Argument(..., help="Start frame image", exists=True),
    end_image: Path = typer.Argument(..., help="End frame image", exists=True),
    motion_prompt: str = typer.Argument(..., help="Natural language motion description"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Base config YAML"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    project_name: str = typer.Option("animation", "--name", "-n", help="Project name"),
    fps: int = typer.Option(24, "--fps", help="Frames per second"),
    frames: int = typer.Option(48, "--frames", help="Total frame count"),
    draft: bool = typer.Option(False, "--draft", help="Draft mode (faster, lower quality)"),
):
    """Create and validate animation configuration."""
    console.print(Panel.fit(
        f"[bold]Nova Animation Config[/bold]\n"
        f"Start: {start_image}\n"
        f"End: {end_image}\n"
        f"Motion: {motion_prompt}\n"
        f"FPS: {fps} | Frames: {frames} | Draft: {draft}",
        title="Configuration",
        border_style="blue",
    ))

    try:
        cfg = create_animation_config(
            start_image=start_image,
            end_image=end_image,
            motion_prompt=motion_prompt,
            config_path=config_file,
            output_dir=output_dir,
            project_name=project_name,
            fps=fps,
            frame_count=frames,
            generation={"draft_mode": draft},
        )
        console.print("[green]✓[/green] Configuration validated successfully")
        console.print(f"Output directory: {cfg.output_dir}")
        console.print(f"Total frames: {cfg.frame_count} @ {cfg.fps}fps = {cfg.frame_count/cfg.fps:.1f}s")
        return cfg
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        raise typer.Exit(1)


@app.command()
def download_models(
    cache_dir: Path = typer.Option(Path("./models"), "--cache", help="Model cache directory"),
    skip: list[str] = typer.Option([], "--skip", help="Models to skip"),
    verify_only: bool = typer.Option(False, "--verify", help="Only verify existing downloads"),
):
    """Download required models from Hugging Face Hub."""
    import subprocess
    script = Path(__file__).parent.parent.parent / "scripts" / "download_models.py"
    cmd = [sys.executable, str(script), "--cache-dir", str(cache_dir)]
    if skip:
        cmd.extend(["--skip", *skip])
    if verify_only:
        cmd.append("--verify-only")
    console.print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


@app.command()
def doctor():
    """Check system requirements and model availability."""
    console.print(Panel.fit("[bold]Nova System Check[/bold]", border_style="blue"))

    # Python version
    console.print(f"Python: {sys.version.split()[0]}")

    # PyTorch
    try:
        import torch
        console.print(f"PyTorch: {torch.__version__}")
        console.print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"CUDA version: {torch.version.cuda}")
            console.print(f"GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        console.print("[red]PyTorch not installed[/red]")

    # Key dependencies
    deps = ["diffusers", "transformers", "cv2", "mediapipe", "librosa", "whisperx", "torchcrepe"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            console.print(f"  {dep}: {version}")
        except ImportError:
            console.print(f"  {dep}: [red]NOT INSTALLED[/red]")

    # Config file
    config_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    if config_path.exists():
        data = load_config(config_path)
        console.print(f"\nBase config loaded: {len(data)} sections")
    else:
        console.print(f"\n[red]Base config not found: {config_path}[/red]")


@app.command()
def init(
    project_dir: Path = typer.Argument(Path("."), help="Project directory"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing"),
):
    """Initialize a new Nova project structure."""
    project_dir = project_dir.resolve()
    dirs = [
        project_dir / "assets" / "references",
        project_dir / "assets" / "audio",
        project_dir / "output",
        project_dir / "configs",
        project_dir / "logs",
    ]
    for d in dirs:
        if d.exists() and not force:
            console.print(f"[yellow]Exists:[/yellow] {d}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Created:[/green] {d}")

    # Copy base config
    src_config = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    dst_config = project_dir / "configs" / "project.yaml"
    if src_config.exists() and (not dst_config.exists() or force):
        import shutil
        shutil.copy2(src_config, dst_config)
        console.print(f"[green]Copied config:[/green] {dst_config}")

    console.print(f"\n[bold]Project initialized at:[/bold] {project_dir}")


if __name__ == "__main__":
    app()