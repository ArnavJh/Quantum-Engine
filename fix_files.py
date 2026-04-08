# ══════════════════════════════════════════════════════════════
#  Run ONCE — fixes ALL Unicode/emoji encoding issues in 02_
#  Place in same folder as 02_live_quantum_system.py
#  Run: python fix_encoding.py
# ══════════════════════════════════════════════════════════════
 
SYMBOL_MAP = {
    # Currency
    "₹"  : "Rs.",
    # Emojis
    "✅" : "[OK]",
    "⚠️" : "[WARN]",
    "❌" : "[ERROR]",
    "🟢" : "[BUY]",
    "🟡" : "[HOLD]",
    "🔴" : "[SELL]",
    "⚛️" : "[Q]",
    "📊" : "[SAVED]",
    "▶"  : ">>",
    # Arrows
    "↑"  : "UP",
    "↓"  : "DOWN",
    "→"  : "->",
    "←"  : "<-",
    # Box drawing characters (the current error)
    "─"  : "-",
    "━"  : "-",
    "│"  : "|",
    "┃"  : "|",
    "┌"  : "+",
    "┐"  : "+",
    "└"  : "+",
    "┘"  : "+",
    "├"  : "+",
    "┤"  : "+",
    "┬"  : "+",
    "┴"  : "+",
    "┼"  : "+",
    "╔"  : "+",
    "╗"  : "+",
    "╚"  : "+",
    "╝"  : "+",
    "═"  : "=",
    "║"  : "|",
    "╠"  : "+",
    "╣"  : "+",
    "╦"  : "+",
    "╩"  : "+",
    "╬"  : "+",
    # Block characters
    "█"  : "#",
    "░"  : "-",
    "▓"  : "#",
    "▒"  : "-",
    # Misc unicode
    "•"  : "*",
    "·"  : ".",
    "…"  : "...",
    "–"  : "-",
    "—"  : "--",
    "\u2500": "-",   # BOX DRAWINGS LIGHT HORIZONTAL
    "\u2502": "|",   # BOX DRAWINGS LIGHT VERTICAL
    "\u250c": "+",
    "\u2510": "+",
    "\u2514": "+",
    "\u2518": "+",
    "\u251c": "+",
    "\u2524": "+",
    "\u252c": "+",
    "\u2534": "+",
    "\u253c": "+",
    "\u2550": "=",
    "\u2551": "|",
    "\u2554": "+",
    "\u2557": "+",
    "\u255a": "+",
    "\u255d": "+",
    "\u2560": "+",
    "\u2563": "+",
    "\u2566": "+",
    "\u2569": "+",
    "\u256c": "+",
    "\u2588": "#",
    "\u2591": "-",
    "\u2592": "-",
    "\u2593": "#",
    "\u25a0": "#",
    "\u2705": "[OK]",
    "\u274c": "[ERROR]",
    "\u26a0": "[WARN]",
    "\u2b50": "*",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2191": "UP",
    "\u2193": "DOWN",
    "\u20b9": "Rs.",
    "\u2014": "--",
    "\u2013": "-",
    "\u2022": "*",
}
 
fname = "02_live_quantum_system.py"
print(f"Fixing {fname}...")
 
try:
    with open(fname, "r", encoding="utf-8") as f:
        content = f.read()
 
    count = 0
    for old, new in SYMBOL_MAP.items():
        if old in content:
            content = content.replace(old, new)
            count += 1
 
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
 
    print(f"  [OK] {count} symbol types replaced")
 
    # Verify no non-ASCII remains in print statements
    issues = []
    for i, line in enumerate(content.split("\n"), 1):
        if "print(" in line:
            for ch in line:
                if ord(ch) > 127:
                    issues.append(
                        f"  Line {i}: U+{ord(ch):04X} '{ch}'")
                    break
 
    if issues:
        print(f"  [WARN] Remaining non-ASCII in print statements:")
        for iss in issues[:10]:
            print(iss)
    else:
        print("  [OK] No non-ASCII characters in print statements")
 
except FileNotFoundError:
    print(f"  [ERROR] {fname} not found in current folder")
    print("  Make sure fix_encoding.py is in the same folder as 02_")
 
print("\nDone. Run: python 02_live_quantum_system.py")
 