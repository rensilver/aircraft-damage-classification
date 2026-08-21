"""Presentation-only CSS for the Streamlit app."""

from __future__ import annotations

CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2.5rem; max-width: 1200px; }

  .adc-hero {
    background: linear-gradient(135deg, #1B2233 0%, #0E1117 60%);
    border: 1px solid #232A3A;
    border-radius: 16px;
    padding: 1.6rem 1.9rem;
    margin-bottom: 1.6rem;
  }
  .adc-hero h1 {
    font-size: 1.75rem;
    font-weight: 650;
    margin: 0 0 .35rem 0;
    letter-spacing: -0.01em;
  }
  .adc-hero p { color: #9AA4B8; margin: 0; font-size: .95rem; }

  .adc-pill {
    display: inline-block;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    padding: .22rem .6rem;
    border-radius: 999px;
    margin-right: .4rem;
  }
  .adc-pill-crack { background: #3A1D22; color: #FF8A8A; border: 1px solid #5A2A32; }
  .adc-pill-dent  { background: #1D2E3A; color: #7CC7FF; border: 1px solid #2A4A5A; }

  .adc-report {
    background: #12161F;
    border: 1px solid #232A3A;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
  }
  .adc-report h2 {
    font-size: 1.02rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #7CC7FF;
    border-bottom: 1px solid #232A3A;
    padding-bottom: .35rem;
    margin-top: 1.4rem;
  }
  .adc-report h2:first-child { margin-top: 0; }

  .adc-note {
    color: #7A8296;
    font-size: .82rem;
    border-left: 2px solid #2A3245;
    padding-left: .7rem;
    margin-top: 1rem;
  }

  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
