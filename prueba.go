package main

import (
    "fmt"
    "strings"
    "os"
    "log"
    "bufio"
)

func check(e error) {
    if e != nil {
        panic(e)
    }
}

func main() {
    var macs []string

    file, err := os.Open("/home/VICOMTECH/msanz/Desktop/mac_vendors.txt")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        words := strings.Fields(scanner.Text())
        macs= append(macs,words[0])
    }

    if err := scanner.Err(); err != nil {
        log.Fatal(err)
    }
    fmt.Println(macs[0])
}

func printSlice(s []string){
    fmt.Printf("len=%d cap%d %v\n",len(s),cap(s),s)
}
