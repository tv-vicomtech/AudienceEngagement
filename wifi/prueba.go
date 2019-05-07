package main

import (
    "time"
    "fmt"
    "strings"
    "regexp"
    "os"
    "log"
    "io"
    "bufio"
)

func check(e error) {
    if e != nil {
        panic(e)
    }
}

func main() {
    t0 := time.Now()
    var mac_list []string
    var macs []string
    var line_str string
    var reg_mac = regexp.MustCompile(`"d":"wifi-([^"]*)`)
    var length_macs int
    file1, err1 := os.Open("/home/VICOMTECH/msanz/Desktop/mac_vendors.txt")
    file2, err2 := os.Open("/home/VICOMTECH/msanz/Desktop/p2_locations.json")
    check(err1)
    defer file1.Close()
    check(err2)
    defer file2.Close()

    scanner_1 := bufio.NewScanner(file1)
    for scanner_1.Scan() {
        words := strings.Fields(scanner_1.Text())
        mac_list = append(mac_list,words[0])
    }

    if err := scanner_1.Err(); err != nil {
        log.Fatal(err)
    }
    scanner_2 := bufio.NewReader(file2)
    for {
        line,_,err := scanner_2.ReadLine()
        if err == io.EOF {
            break
        }
        line_str = fmt.Sprintf("%s",line)
        words := reg_mac.FindAllString(line_str, -1)
        length_macs=len(words)
        for ii:=0;ii<(length_macs-1);ii++{
            if err == io.EOF {
                break
            }
            s := fmt.Sprintf(words[ii][10:12]+words[ii][13:15]+words[ii][16:18])
            for jj:=0;(jj<len(mac_list)-1);jj++{
                if(strings.Compare(s,mac_list[jj])==0){
                    macs= append(macs,s)
                }
            }
        }
        }
    fmt.Println(macs)
    t1 := time.Now()
    fmt.Println("Time: ", t1.Sub(t0))
}
