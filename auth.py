import sqlite3, bcrypt, os

DB = "kkd.db"

def baglan():
    return sqlite3.connect(DB)

def kur():
    con = baglan()
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS alarms (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        camera TEXT,
        equipment TEXT,
        confidence REAL,
        created_at TEXT NOT NULL,
        image_path TEXT)""")
    con.commit()
    con.close()

def kayit_ol(kullanici, sifre):
    con = baglan()
    try:
        h = bcrypt.hashpw(sifre.encode(), bcrypt.gensalt()).decode()
        con.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (kullanici, h))
        con.commit()
        return True, "Kayit basarili"
    except sqlite3.IntegrityError:
        return False, "Bu kullanici adi zaten alinmis"
    finally:
        con.close()

def giris_yap(kullanici, sifre):
    con = baglan()
    row = con.execute("SELECT password_hash FROM users WHERE username = ?",
                      (kullanici,)).fetchone()
    con.close()
    if row and bcrypt.checkpw(sifre.encode(), row[0].encode()):
        return True
    return False

def alarm_kaydet(kullanici, kamera, ekipman, guven, tarih, resim):
    con = baglan()
    con.execute("""INSERT INTO alarms
        (username, camera, equipment, confidence, created_at, image_path)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (kullanici, kamera, ekipman, guven, tarih, resim))
    con.commit()
    con.close()

def alarmlari_getir(kullanici):
    con = baglan()
    rows = con.execute("""SELECT camera, equipment, confidence, created_at, image_path
        FROM alarms WHERE username = ? ORDER BY created_at DESC""",
        (kullanici,)).fetchall()
    con.close()
    return rows

kur()