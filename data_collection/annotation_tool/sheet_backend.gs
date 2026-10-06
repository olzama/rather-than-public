/**
 * Google Apps Script backend for the "rather than" annotator.
 *
 * Stores every label as a row in the sheet "labels" (created on first use):
 *   received_at | annotator | item_id | label | client_ts | set
 * A label is never overwritten; the latest row for an annotator and item wins.
 *
 * Endpoints (deploy as a web app, "Execute as: Me", "Who has access: Anyone"):
 *   GET  ?token=T&annotator=NAME           -> {"answers": {item_id: label, ...}}
 *   GET  ?token=T&annotator=NAME&ts=1      -> also {"ts": {item_id: client_ts, ...}}
 *   POST {"token":T,"action":"save", "annotator","item_id","label","ts","set"}
 *   POST {"token":T,"action":"import","rows":[{annotator,item_id,label,ts,set}, ...]}
 * POST bodies are sent as text/plain so browsers skip the CORS preflight.
 *
 * TOKEN only keeps out stray requests; it is visible in the page source.
 */
var TOKEN = "rt-2026-annotate";
var SHEET = "labels";
var HEADER = ["received_at", "annotator", "item_id", "label", "client_ts", "set"];

function sheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(SHEET);
  if (!sh) {
    sh = ss.insertSheet(SHEET);
    sh.appendRow(HEADER);
    sh.setFrozenRows(1);
  }
  return sh;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doGet(e) {
  var p = e.parameter || {};
  if (p.token !== TOKEN) return json_({ error: "bad token" });
  var name = String(p.annotator || "").trim();
  if (!name) return json_({ error: "missing annotator" });
  var rows = sheet_().getDataRange().getValues();
  var answers = {}, ts = {};
  for (var i = 1; i < rows.length; i++) {
    if (String(rows[i][1]).trim().toLowerCase() === name.toLowerCase()) {
      answers[rows[i][2]] = rows[i][3];
      ts[rows[i][2]] = rows[i][4];
    }
  }
  return json_(p.ts ? { answers: answers, ts: ts } : { answers: answers });
}

function doPost(e) {
  var body;
  try { body = JSON.parse(e.postData.contents); } catch (err) { return json_({ error: "bad json" }); }
  if (body.token !== TOKEN) return json_({ error: "bad token" });
  var rows = [];
  var now = new Date().toISOString();
  var add = function (r) {
    if (!r || !r.annotator || !r.item_id || !r.label) return;
    rows.push([now, String(r.annotator).trim(), String(r.item_id), String(r.label), r.ts || "", r.set || "main"]);
  };
  if (body.action === "save") add(body);
  else if (body.action === "import") (body.rows || []).forEach(add);
  else return json_({ error: "unknown action" });
  if (!rows.length) return json_({ error: "nothing to write" });
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var sh = sheet_();
    sh.getRange(sh.getLastRow() + 1, 1, rows.length, HEADER.length).setValues(rows);
  } finally {
    lock.releaseLock();
  }
  return json_({ ok: true, written: rows.length });
}
