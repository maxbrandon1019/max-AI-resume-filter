"""Pure Python SVG Radar and Metric Chart Generators for Streamlit UI."""
import math
from typing import Dict, Optional, List, Tuple


def _polar_to_cartesian(cx: float, cy: float, radius: float, angle_rad: float) -> Tuple[float, float]:
    """Converts polar coordinates (angle, radius) to Cartesian coordinates (x, y)."""
    return cx + radius * math.cos(angle_rad), cy + radius * math.sin(angle_rad)


def generate_radar_chart_svg(
    candidate_name: str,
    scores: Dict[str, float],
    benchmark: Optional[Dict[str, float]] = None,
    width: int = 440,
    height: int = 380
) -> str:
    """
    Generates a theme-adaptive, high-DPI SVG radar chart for a single candidate.
    Zero external dependencies (pure math + SVG).
    """
    cx, cy = width / 2.0, height / 2.0 - 10
    max_radius = min(cx, cy) - 55

    axes = list(scores.keys())
    n = len(axes)
    if n < 3:
        return "<p>Not enough dimensions for radar chart</p>"

    # Ring levels
    rings = [0.2, 0.4, 0.6, 0.8, 1.0]

    svg_parts: List[str] = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="max-width: {width}px; margin: auto; display: block; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">',
        '<defs>',
        '  <linearGradient id="gradCandidate" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#4F46E5" stop-opacity="0.5"/>',
        '    <stop offset="100%" stop-color="#06B6D4" stop-opacity="0.3"/>',
        '  </linearGradient>',
        '  <linearGradient id="gradBenchmark" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#94A3B8" stop-opacity="0.2"/>',
        '    <stop offset="100%" stop-color="#64748B" stop-opacity="0.1"/>',
        '  </linearGradient>',
        '  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '    <feGaussianBlur stdDeviation="3" result="blur" />',
        '    <feComposite in="SourceGraphic" in2="blur" operator="over" />',
        '  </filter>',
        '</defs>'
    ]

    # Concentric background polygon rings
    for r_level in rings:
        r = max_radius * r_level
        ring_pts = []
        for i in range(n):
            angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
            px, py = _polar_to_cartesian(cx, cy, r, angle)
            ring_pts.append(f"{px:.1f},{py:.1f}")
        opacity = "0.35" if r_level == 1.0 else "0.15"
        stroke_dash = "" if r_level == 1.0 else 'stroke-dasharray="3,3"'
        svg_parts.append(f'<polygon points="{" ".join(ring_pts)}" fill="none" stroke="currentColor" stroke-opacity="{opacity}" stroke-width="1" {stroke_dash}/>')
        # Label 100% on top axis
        if r_level in {0.5, 1.0}:
            label_y = cy - r - 2
            svg_parts.append(f'<text x="{cx + 4:.1f}" y="{label_y:.1f}" font-size="9" fill="currentColor" opacity="0.45">{int(r_level * 100)}%</text>')

    # Axis spokes & labels
    for i, axis in enumerate(axes):
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        end_x, end_y = _polar_to_cartesian(cx, cy, max_radius, angle)
        svg_parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>')

        # Label position
        label_r = max_radius + 24
        lx, ly = _polar_to_cartesian(cx, cy, label_r, angle)
        score_val = scores.get(axis, 0.0)

        anchor = "middle"
        if math.cos(angle) > 0.25:
            anchor = "start"
        elif math.cos(angle) < -0.25:
            anchor = "end"

        # Multi-line label for readability
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly - 5:.1f}" text-anchor="{anchor}" font-size="11" font-weight="600" fill="currentColor" opacity="0.85">{axis}</text>'
        )
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly + 9:.1f}" text-anchor="{anchor}" font-size="11" font-weight="700" fill="#4F46E5">{score_val:.0f}%</text>'
        )

    # Benchmark polygon (if provided)
    if benchmark:
        bench_pts = []
        for i, axis in enumerate(axes):
            angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
            val = min(100.0, max(0.0, benchmark.get(axis, 0.0)))
            r = max_radius * (val / 100.0)
            bx, by = _polar_to_cartesian(cx, cy, r, angle)
            bench_pts.append(f"{bx:.1f},{by:.1f}")
        svg_parts.append(f'<polygon points="{" ".join(bench_pts)}" fill="url(#gradBenchmark)" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="4,3"/>')

    # Candidate polygon
    cand_pts = []
    coord_list = []
    for i, axis in enumerate(axes):
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        val = min(100.0, max(0.0, scores.get(axis, 0.0)))
        r = max_radius * (val / 100.0)
        px, py = _polar_to_cartesian(cx, cy, r, angle)
        cand_pts.append(f"{px:.1f},{py:.1f}")
        coord_list.append((px, py, axis, val))

    svg_parts.append(f'<polygon points="{" ".join(cand_pts)}" fill="url(#gradCandidate)" stroke="#4F46E5" stroke-width="2.5" filter="url(#glow)"/>')

    # Data points
    for px, py, axis, val in coord_list:
        svg_parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#4F46E5" stroke="#FFFFFF" stroke-width="1.5"/>')

    # Legend / Title
    svg_parts.append(f'<text x="{cx:.1f}" y="{height - 10:.1f}" text-anchor="middle" font-size="11" font-weight="500" fill="currentColor" opacity="0.6">Target Fit Profile: <tspan font-weight="700" fill="#4F46E5">{candidate_name}</tspan></text>')
    svg_parts.append('</svg>')

    return "\n".join(svg_parts)


def generate_comparison_radar_chart_svg(
    name_a: str,
    scores_a: Dict[str, float],
    name_b: str,
    scores_b: Dict[str, float],
    width: int = 480,
    height: int = 400
) -> str:
    """
    Generates a dual-candidate head-to-head SVG radar comparison chart.
    Candidate A: Royal Indigo / Blue (#3B82F6)
    Candidate B: Emerald / Teal (#10B981)
    """
    cx, cy = width / 2.0, height / 2.0 - 15
    max_radius = min(cx, cy) - 60

    axes = list(scores_a.keys())
    n = len(axes)
    if n < 3:
        return "<p>Not enough dimensions for radar chart</p>"

    rings = [0.2, 0.4, 0.6, 0.8, 1.0]

    svg_parts: List[str] = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="max-width: {width}px; margin: auto; display: block; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">',
        '<defs>',
        '  <linearGradient id="gradCandA" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#3B82F6" stop-opacity="0.45"/>',
        '    <stop offset="100%" stop-color="#1D4ED8" stop-opacity="0.25"/>',
        '  </linearGradient>',
        '  <linearGradient id="gradCandB" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#10B981" stop-opacity="0.45"/>',
        '    <stop offset="100%" stop-color="#047857" stop-opacity="0.25"/>',
        '  </linearGradient>',
        '</defs>'
    ]

    # Concentric rings
    for r_level in rings:
        r = max_radius * r_level
        ring_pts = []
        for i in range(n):
            angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
            px, py = _polar_to_cartesian(cx, cy, r, angle)
            ring_pts.append(f"{px:.1f},{py:.1f}")
        opacity = "0.35" if r_level == 1.0 else "0.15"
        stroke_dash = "" if r_level == 1.0 else 'stroke-dasharray="3,3"'
        svg_parts.append(f'<polygon points="{" ".join(ring_pts)}" fill="none" stroke="currentColor" stroke-opacity="{opacity}" stroke-width="1" {stroke_dash}/>')
        if r_level in {0.5, 1.0}:
            svg_parts.append(f'<text x="{cx + 4:.1f}" y="{cy - r - 2:.1f}" font-size="9" fill="currentColor" opacity="0.45">{int(r_level * 100)}%</text>')

    # Spokes & Axis Labels
    for i, axis in enumerate(axes):
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        end_x, end_y = _polar_to_cartesian(cx, cy, max_radius, angle)
        svg_parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>')

        label_r = max_radius + 26
        lx, ly = _polar_to_cartesian(cx, cy, label_r, angle)
        val_a = scores_a.get(axis, 0.0)
        val_b = scores_b.get(axis, 0.0)

        anchor = "middle"
        if math.cos(angle) > 0.25:
            anchor = "start"
        elif math.cos(angle) < -0.25:
            anchor = "end"

        svg_parts.append(f'<text x="{lx:.1f}" y="{ly - 7:.1f}" text-anchor="{anchor}" font-size="11" font-weight="600" fill="currentColor" opacity="0.85">{axis}</text>')
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly + 7:.1f}" text-anchor="{anchor}" font-size="10" font-weight="700">'
            f'<tspan fill="#3B82F6">{val_a:.0f}%</tspan> <tspan fill="currentColor" opacity="0.4">vs</tspan> <tspan fill="#10B981">{val_b:.0f}%</tspan>'
            f'</text>'
        )

    # Candidate A polygon (Blue)
    pts_a = []
    coords_a = []
    for i, axis in enumerate(axes):
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        val = min(100.0, max(0.0, scores_a.get(axis, 0.0)))
        r = max_radius * (val / 100.0)
        px, py = _polar_to_cartesian(cx, cy, r, angle)
        pts_a.append(f"{px:.1f},{py:.1f}")
        coords_a.append((px, py))

    svg_parts.append(f'<polygon points="{" ".join(pts_a)}" fill="url(#gradCandA)" stroke="#3B82F6" stroke-width="2.2"/>')

    # Candidate B polygon (Emerald)
    pts_b = []
    coords_b = []
    for i, axis in enumerate(axes):
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        val = min(100.0, max(0.0, scores_b.get(axis, 0.0)))
        r = max_radius * (val / 100.0)
        px, py = _polar_to_cartesian(cx, cy, r, angle)
        pts_b.append(f"{px:.1f},{py:.1f}")
        coords_b.append((px, py))

    svg_parts.append(f'<polygon points="{" ".join(pts_b)}" fill="url(#gradCandB)" stroke="#10B981" stroke-width="2.2"/>')

    # Points for A and B
    for px, py in coords_a:
        svg_parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#3B82F6" stroke="#FFFFFF" stroke-width="1.5"/>')
    for px, py in coords_b:
        svg_parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#10B981" stroke="#FFFFFF" stroke-width="1.5"/>')

    # Interactive Legend
    svg_parts.append(
        f'<g transform="translate({cx - 130:.1f}, {height - 18:.1f})">'
        f'  <circle cx="10" cy="5" r="5" fill="#3B82F6"/>'
        f'  <text x="20" y="9" font-size="11" font-weight="600" fill="#3B82F6">{name_a[:16]}</text>'
        f'  <circle cx="140" cy="5" r="5" fill="#10B981"/>'
        f'  <text x="150" y="9" font-size="11" font-weight="600" fill="#10B981">{name_b[:16]}</text>'
        f'</g>'
    )
    svg_parts.append('</svg>')

    return "\n".join(svg_parts)
