// sign: release signing for the Windows worker (run by the windows-worker workflow; needs the repo secret
// KOVAPOOL_SIGN_KEY = an Ed25519 private key in PKCS#8 PEM, made once with `openssl genpkey -algorithm ed25519`).
//
//	go run ./tools/sign -pub                 prints the base64 public key (KovaPool.exe embeds it: -X main.signPub=...)
//	go run ./tools/sign FILE [FILE...]       writes SHA256SUMS (sha256sum format) and SHA256SUMS.sig (base64 Ed25519
//	                                         signature over the bytes of SHA256SUMS) into the first file's directory
//
// KovaPool.exe installs a kangaroo-windows.zip only when SHA256SUMS verifies against its embedded key and lists the
// zip's hash, so the rolling release can be replaced by nobody who lacks the key.
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func die(f string, a ...any) { fmt.Fprintf(os.Stderr, "sign: "+f+"\n", a...); os.Exit(1) }

func key() ed25519.PrivateKey {
	blk, _ := pem.Decode([]byte(os.Getenv("KOVAPOOL_SIGN_KEY")))
	if blk == nil {
		die("KOVAPOOL_SIGN_KEY is not a PEM key (openssl genpkey -algorithm ed25519 -out kovapool-sign.pem)")
	}
	k, err := x509.ParsePKCS8PrivateKey(blk.Bytes)
	if err != nil {
		die("KOVAPOOL_SIGN_KEY: %v", err)
	}
	priv, ok := k.(ed25519.PrivateKey)
	if !ok {
		die("KOVAPOOL_SIGN_KEY is not an Ed25519 key")
	}
	return priv
}

func main() {
	if len(os.Args) < 2 {
		die("usage: sign -pub | sign FILE [FILE...]")
	}
	priv := key()
	if os.Args[1] == "-pub" {
		fmt.Println(base64.StdEncoding.EncodeToString(priv.Public().(ed25519.PublicKey)))
		return
	}
	var sums string
	for _, path := range os.Args[1:] {
		f, err := os.Open(path)
		if err != nil {
			die("%v", err)
		}
		h := sha256.New()
		if _, err := io.Copy(h, f); err != nil {
			die("%v", err)
		}
		f.Close()
		sums += fmt.Sprintf("%s  %s\n", hex.EncodeToString(h.Sum(nil)), filepath.Base(path))
	}
	dir := filepath.Dir(os.Args[1])
	if err := os.WriteFile(filepath.Join(dir, "SHA256SUMS"), []byte(sums), 0o644); err != nil {
		die("%v", err)
	}
	sig := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, []byte(sums)))
	if err := os.WriteFile(filepath.Join(dir, "SHA256SUMS.sig"), []byte(sig+"\n"), 0o644); err != nil {
		die("%v", err)
	}
	fmt.Print(sums)
}
