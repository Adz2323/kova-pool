#!/usr/bin/env python3
"""
payout_electrum.py: turn the pool's share ledger into an Electrum "Pay to many" list after a find.

Nothing here signs, broadcasts or touches a key. It reads the ledger (from the live pool with the admin password, or
straight from pool.db on the pool host), applies the published formula
    payout = (prize - network fee - pool fee) x account shares / all shares        (pool fee = FEE_PERCENT of the prize)
and writes three files you check by eye before anything moves:

    payout_<sid>.electrum.txt   one "address,amount" line per paid account: in Electrum, Tools > Pay to many, paste
                                the file's contents, set the fee, review, sign.
    payout_<sid>.ledger.json    the public ledger: every account's shares, fraction, amount and status (paid / held /
                                forfeited), labelled the way the board labels them. No emails. Publish this with the txid.
    payout_<sid>.summary.txt    what you are about to do, in words, with the totals that must add up.

From your own machine (the usual way; asks for the admin password, or reads ADMIN_PASSWORD from the environment):
    python3 payout_electrum.py --url https://pool.kova23.com --received 13.9985 --prize 14
On the pool host, from the database directly:
    python3 payout_electrum.py --db pool.db --received 13.9985 --prize 14

    --received    BTC that actually arrived in the pool wallet from the sweep (the sweep's own network fee is already gone)
    --prize       BTC the puzzle address held before the sweep; the pool fee is FEE_PERCENT of this (default: --received)
    --payout-fee  BTC reserved for the payout transaction's network fee (estimate; set Electrum's fee at or below it, the
                  unspent remainder stays in the wallet). Default 0.0005.
    --sid         the search the find belongs to; default: the find recorded by pool.py when the key was derived
    --until       last UTC day of shares that count (inclusive); default: the day of the find
    --fee-percent default: the pool's FEE_PERCENT (from the server, or pool.env next to pool.db, else 15)
    --dust        amounts below this are forfeited, as the terms say (default 0.0001)
    --out         directory for the three files (default: current directory)

Rules applied (terms section 7 and the How page):
  - solo find: the finder's account receives everything after the pool fee; pooled accounts receive nothing from it
  - pooled find: split by shares of this search up to and including the find day; solo tokens have no shares
  - a token that is suspended ("DP failed verification") has its shares HELD for review, not paid, not forfeited, and
    those shares are not in anyone's denominator, so the review never changes an honest member's cut
  - an account without a payout address is HELD (30 days' notice by email, per the terms), its amount stays in the wallet
  - amounts under the dust threshold are forfeited and stay in the wallet
Amounts are computed in satoshis with floor rounding so the lines never add up to more than the distributable amount.
Only the Python standard library.
"""
import argparse, getpass, hashlib, json, os, re, sqlite3, sys, time, urllib.error, urllib.parse, urllib.request
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

SAT = Decimal(100_000_000)
ADDR = re.compile(r"^(1[1-9A-HJ-NP-Za-km-z]{25,34}|3[1-9A-HJ-NP-Za-km-z]{25,34}|bc1[02-9ac-hj-np-z]{11,87})$")

def btc(x): return f"{(Decimal(x) / SAT).quantize(Decimal('0.00000001')):.8f}"
def sats(x): return int((Decimal(str(x)) * SAT).to_integral_value(rounding=ROUND_DOWN))

# ------------------------------------------------------------------ the ledger, from the live pool or from pool.db
def ledger_from_url(url, sid, until):
    pw = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("pool admin password: ")
    q = urllib.parse.urlencode({k: v for k, v in (("sid", sid), ("until", until)) if v})
    req = urllib.request.Request(url.rstrip("/") + "/api/admin/ledger" + ("?" + q if q else ""), headers={"X-Admin-Password": pw})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"pool said {e.code}: {e.read().decode(errors='replace')[:200]}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {url}: {e.reason}")

def ledger_from_db(path, sid, until):
    if not Path(path).exists(): sys.exit(f"no database at {path} (use --db or --url)")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True); db.row_factory = sqlite3.Row
    S = lambda k, d="": (db.execute("SELECT v FROM setting WHERE k=?", (k,)).fetchone() or {"v": d})["v"]
    sid = sid or S("finder_sid")
    if not sid: sys.exit("no find recorded (setting finder_sid is empty) and no --sid given")
    ft = float(S("finder_t", "0") or 0)
    until = until or (time.strftime("%Y-%m-%d", time.gmtime(ft)) if ft else time.strftime("%Y-%m-%d", time.gmtime()))
    rows = db.execute("""SELECT a.id, a.display, a.payout, a.banned,
                                COALESCE(SUM(CASE WHEN t.banned='' THEN s.shares END),0) ok,
                                COALESCE(SUM(CASE WHEN t.banned<>'' THEN s.shares END),0) held_sh
                         FROM shares_daily s JOIN token t ON t.id=s.token JOIN account a ON a.id=t.account
                         WHERE s.sid=? AND s.day<=? AND t.mode='pooled' GROUP BY a.id""", (sid, until)).fetchall()
    finder = S("finder"); fa = None
    if finder:
        r = db.execute("SELECT account FROM token WHERE id=?", (finder,)).fetchone(); fa = r["account"] if r else None
    fr = db.execute("SELECT id, display, payout, banned FROM account WHERE id=?", (fa,)).fetchone() if fa is not None else None
    fee = None
    envf = Path(path).parent / "pool.env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("FEE_PERCENT="): fee = float(line.split("=", 1)[1].split("#")[0].strip())
    return {"sid": sid, "until": until, "finder_token": finder, "finder_mode": S("finder_mode", "pooled"), "finder_t": ft or None,
            "finder_account": fa, "finder_row": dict(fr) if fr else None, "fee_percent": fee, "board_salt": S("board_salt"),
            "pool_shares": float(S("pool_shares:" + sid, "0") or 0), "rows": [dict(r) for r in rows]}

# ------------------------------------------------------------------ the split
def main():
    ap = argparse.ArgumentParser(description="share ledger -> Electrum pay-to-many list")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", help="pool site, e.g. https://pool.kova23.com (asks for the admin password)")
    src.add_argument("--db", help="pool.db on the pool host (default when neither is given: $DB_FILE or ./pool.db)")
    ap.add_argument("--received", required=True, help="BTC that arrived from the sweep")
    ap.add_argument("--prize", help="BTC the address held before the sweep (default: --received)")
    ap.add_argument("--payout-fee", default="0.0005", help="BTC reserved for the payout tx fee")
    ap.add_argument("--sid"); ap.add_argument("--until"); ap.add_argument("--fee-percent", type=float)
    ap.add_argument("--dust", default="0.0001"); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.until and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.until): sys.exit("--until must be YYYY-MM-DD")

    L = ledger_from_url(a.url, a.sid, a.until) if a.url else ledger_from_db(a.db or os.environ.get("DB_FILE", "pool.db"), a.sid, a.until)
    sid, until, finder_mode, finder_t, finder_acct = L["sid"], L["until"], L["finder_mode"] or "pooled", L["finder_t"] or 0, L["finder_account"]
    if not sid: sys.exit("no find recorded and no --sid given")
    fee_pct = a.fee_percent if a.fee_percent is not None else (L["fee_percent"] if L["fee_percent"] is not None else 15.0)

    received = sats(a.received); prize = sats(a.prize) if a.prize else received
    pool_fee = int(Decimal(prize) * Decimal(str(fee_pct)) / 100)
    reserve = sats(a.payout_fee)
    distributable = received - pool_fee - reserve
    if distributable <= 0: sys.exit(f"nothing to distribute: received {btc(received)} - pool fee {btc(pool_fee)} - fee reserve {btc(reserve)} <= 0")
    dust = sats(a.dust)
    salt = L["board_salt"] or ""
    def label(r):
        if r.get("display"): return r["display"]
        if r.get("payout"): return f"{r['payout'][:6]}…{r['payout'][-4:]}"
        return f"member-{hashlib.sha1(f'{salt}:{r['id']}'.encode()).hexdigest()[:4]}"

    entries = []
    if finder_mode == "solo":
        fr = L["finder_row"]
        if not fr: sys.exit("finder_mode is solo but the finder's account could not be found")
        entries.append({"account": fr["id"], "label": label(fr), "shares": 0.0, "fraction": 1.0, "sats": distributable,
                        "address": fr.get("payout") or "", "finder": True, "banned": bool(fr.get("banned")), "held_shares": 0.0})
        total_sh = 0.0
    else:
        rows = L["rows"]; total_sh = sum(r["ok"] for r in rows)
        if total_sh <= 0: sys.exit(f"no pooled shares for search {sid} up to {until}")
        for r in rows:
            frac = Decimal(r["ok"]) / Decimal(total_sh)
            entries.append({"account": r["id"], "label": label(r), "shares": r["ok"], "fraction": float(frac),
                            "sats": int(Decimal(distributable) * frac), "address": r.get("payout") or "",
                            "finder": r["id"] == finder_acct, "banned": bool(r.get("banned")), "held_shares": r["held_sh"]})
    entries.sort(key=lambda e: -e["sats"])

    paid, held, forfeited = [], [], []
    for e in entries:
        if e["banned"] or (e["held_shares"] and not e["shares"]): e["status"] = "held: account or token under review"; held.append(e)
        elif not e["address"] or not ADDR.match(e["address"]): e["status"] = "held: no verified payout address (30 days' notice)"; held.append(e)
        elif e["sats"] < dust: e["status"] = "forfeited: under the dust threshold"; forfeited.append(e)
        else: e["status"] = "paid"; paid.append(e)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f_el = out / f"payout_{sid}.electrum.txt"; f_lg = out / f"payout_{sid}.ledger.json"; f_sm = out / f"payout_{sid}.summary.txt"
    f_el.write_text("".join(f"{e['address']},{btc(e['sats'])}\n" for e in paid))
    ledger = {"search": sid, "find_at": finder_t or None, "paid_as": finder_mode, "shares_until_day": until,
              "prize_btc": btc(prize), "received_btc": btc(received), "pool_fee_percent": fee_pct, "pool_fee_btc": btc(pool_fee),
              "payout_fee_reserve_btc": btc(reserve), "distributable_btc": btc(distributable), "dust_btc": btc(dust),
              "total_shares": total_sh, "pool_shares_setting": L["pool_shares"],
              "accounts": [{"label": e["label"], "shares": e["shares"], "fraction": e["fraction"], "amount_btc": btc(e["sats"]),
                            "address": e["address"] if e["status"] == "paid" else "", "finder": e["finder"], "status": e["status"],
                            "shares_under_review": e["held_shares"]} for e in entries],
              "txids": [], "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    f_lg.write_text(json.dumps(ledger, indent=1))

    s_paid = sum(e["sats"] for e in paid); s_held = sum(e["sats"] for e in held); s_forf = sum(e["sats"] for e in forfeited)
    lines = [f"Payout for search {sid}, paid as {finder_mode}" + (f", key derived {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(finder_t))}" if finder_t else ""),
             f"shares counted up to and including {until} (UTC); ledger from {a.url or 'the database'}",
             f"prize {btc(prize)} BTC, received {btc(received)}, pool fee {fee_pct:g}% = {btc(pool_fee)}, fee reserve {btc(reserve)}",
             f"distributable {btc(distributable)} BTC over {total_sh:.6g} shares" + (f" (pool_shares setting says {L['pool_shares']:.6g}; a gap means shares after the find day)" if finder_mode != "solo" else ""),
             "",
             f"PAID      {len(paid):4d} accounts  {btc(s_paid)} BTC  -> {f_el.name}",
             f"HELD      {len(held):4d} accounts  {btc(s_held)} BTC  (stays in the wallet; pay by hand when resolved)",
             f"FORFEITED {len(forfeited):4d} accounts  {btc(s_forf)} BTC  (under {btc(dust)}, stays in the wallet)",
             f"remainder {btc(distributable - s_paid - s_held - s_forf)} BTC from floor rounding also stays in the wallet",
             "", "Largest ten:"]
    for e in entries[:10]: lines.append(f"  {e['label']:<24} {e['fraction']*100:7.3f}%  {btc(e['sats'])} BTC  {e['status']}" + ("  (finder)" if e["finder"] else ""))
    if held: lines += ["", "Held:"] + [f"  {e['label']:<24} {btc(e['sats'])} BTC  {e['status']}" for e in held]
    lines += ["", "Next: open Electrum, Tools > Pay to many, paste the contents of " + f_el.name + ", set the fee at or below the reserve,",
              "check the total against PAID above, sign and broadcast. Then put the txid into " + f_lg.name + " (txids) and publish it."]
    f_sm.write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print(f"\nwrote {f_el}\n      {f_lg}\n      {f_sm}")

if __name__ == "__main__": main()
