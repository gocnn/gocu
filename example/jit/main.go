package main

import _ "embed"

//go:embed add.ptx
var ptx string

func main() {
	println(ptx)
}
