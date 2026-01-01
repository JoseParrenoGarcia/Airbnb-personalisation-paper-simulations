from pathlib import Path
import pandas as pd

def save_pretty_table_html(
    df: pd.DataFrame,
    path: str = "table_preview.html",
    caption: str | None = None,
    footnote: str | None = None,
    float_format: str = "{:.3f}",
    rename_cols: dict | None = None,
    emphasise_col: str | None = None,   # column name AFTER renaming
    max_width_px: int = 1000,
):
    """
    Save a minimalist, iOS-style HTML table for PyCharm/browser viewing.

    Features:
    - reader-friendly column naming
    - centred cells (headers + body)
    - subtle emphasis on one key column
    - optional caption + footnote
    """

    df_fmt = df.copy()

    # 1) Rename columns for readability
    if rename_cols:
        df_fmt = df_fmt.rename(columns=rename_cols)

    # 2) Format floats
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            if pd.api.types.is_integer_dtype(df_fmt[col]):
                continue
            df_fmt[col] = df_fmt[col].map(
                lambda x: float_format.format(x) if pd.notna(x) else ""
            )

    # Build HTML table
    html_table = df_fmt.to_html(
        index=False,
        escape=False,
        classes="pretty-table",
    )

    caption_html = f"<div class='table-caption'>{caption}</div>" if caption else ""
    footnote_html = f"<div class='table-footnote'>{footnote}</div>" if footnote else ""

    # Identify emphasised column index (1-based nth-child in CSS)
    emphasise_css = ""
    if emphasise_col:
        cols = list(df_fmt.columns)
        if emphasise_col not in cols:
            raise ValueError(f"emphasise_col='{emphasise_col}' not found in columns: {cols}")
        emphasise_idx = cols.index(emphasise_col) + 1

        emphasise_css = f"""
        table.pretty-table td:nth-child({emphasise_idx}),
        table.pretty-table th:nth-child({emphasise_idx}) {{
            font-weight: 600;
            background: #fbfbfd;
        }}
        """

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body {{
            background: #ffffff;
            margin: 24px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;
            color: #1c1c1e;
          }}

          .table-container {{
            max-width: {max_width_px}px;
            margin: 24px auto;
          }}

          .table-caption {{
            font-size: 20px;
            font-weight: 650;
            margin-bottom: 14px;
            color: #1c1c1e;
            text-align: center;
          }}

          table.pretty-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
          }}

          table.pretty-table thead th {{
            background: #f5f5f7;
            font-weight: 650;
            padding: 14px 16px;
            font-size: 14px;
            border-bottom: 2px solid #d1d1d6;
            color: #1c1c1e;
            text-align: center;
          }}

          table.pretty-table tbody td {{
            padding: 14px 16px;
            font-size: 14px;
            border-bottom: 1px solid #e5e5ea;
            color: #1c1c1e;
            text-align: center;
          }}

          table.pretty-table tbody tr:last-child td {{
            border-bottom: none;
          }}

          table.pretty-table tbody tr:hover {{
            background-color: #f9f9fb;
          }}

          {emphasise_css}

          .table-footnote {{
            margin-top: 10px;
            font-size: 12px;
            line-height: 1.4;
            color: #6e6e73;
            text-align: center;
          }}
        </style>
      </head>
      <body>
        <div class="table-container">
          {caption_html}
          {html_table}
          {footnote_html}
        </div>
      </body>
    </html>
    """

    out = Path(path).resolve()
    out.write_text(html, encoding="utf-8")
    return str(out)
