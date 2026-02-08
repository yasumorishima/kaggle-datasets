#!/usr/bin/env python3
import re, sys, json

def parse_md(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r'###\s+\*\*([^*]+)\*\*\s+```\s+([^`]+)```'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    return {name.strip(): desc.strip() for name, desc in matches}

def generate_js(columns):
    columns_json = json.dumps(columns, indent=2, ensure_ascii=False)
    return f"""// Kaggle Column Description Auto-Filler
(function() {{
    const columns = {columns_json};
    let updated = 0, skipped = 0, failed = 0;
    const headers = document.querySelectorAll('span[title]');
    console.log('Found ' + headers.length + ' columns');

    headers.forEach((header) => {{
        const name = header.getAttribute('title');
        if (!columns[name]) {{ skipped++; return; }}
        try {{
            const th = header.closest('th');
            const input = th.querySelector('input[placeholder="Please enter a description"]');
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(input, columns[name]);
            input.dispatchEvent(new Event('input', {{bubbles: true}}));
            input.dispatchEvent(new Event('change', {{bubbles: true}}));
            input.dispatchEvent(new Event('blur', {{bubbles: true}}));
            console.log('[OK] ' + name);
            updated++;
        }} catch(e) {{
            console.error('[ERROR] ' + name + ': ' + e.message);
            failed++;
        }}
    }});

    console.log('\\n=== SUMMARY ===');
    console.log('Updated: ' + updated);
    console.log('Skipped: ' + skipped);
    console.log('Failed: ' + failed);
    console.log('\\n[IMPORTANT] Please review and click SAVE!');
}})();
"""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_js.py <markdown_file>")
        sys.exit(1)
    columns = parse_md(sys.argv[1])
    print(generate_js(columns))
