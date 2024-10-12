package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

// Earth radius in km
const R = 6378.137

// grid unit in m
var gridUnit = 1.0

const JS_HEAD = `(function(){
	var points = [
`

const JS_TAIL = `];
})();
`

func degree2radian(degree float64) float64 {
	return degree * math.Pi / 180
}

// normalize x into rounded value
func Round(x, unit float64) float64 {
	return math.Round(x/unit) * unit
}

// Coordinate definition
type Coordinate struct {
	lat float64
	lon float64
}

// read file line by line
func ReadFile(path string, callback func(string)) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	for {
		if scanner.Scan() {
			// scanner.Text() returns string, scanner.Bytes() returns []byte
			line := scanner.Text()

			// process this line
			callback(line)
			continue
		}
		break
	}

	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

// ファイルを読んで統計情報を記録するための構造体
type Stats struct {
	lineErrors []error
	linenum    int

	minLat, maxLat float64
	minLon, maxLon float64
	maxDepth       float64

	distanceWestEast   int
	distanceSouthNorth int

	gridLat float64
	gridLon float64

	// 値は水深
	data map[Coordinate]float64
}

func NewStats() *Stats {
	stats := &Stats{}

	stats.minLat = 90.0
	stats.minLon = 180.0

	stats.data = make(map[Coordinate]float64)

	return stats
}

func (s *Stats) Print() *Stats {
	fmt.Println(s)
	return s
}

func (s *Stats) String() string {
	var out bytes.Buffer

	out.WriteString(fmt.Sprintf("parsed lines: %d", s.linenum))
	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("parse errors: %d", len(s.lineErrors)))
	out.WriteString("\n")

	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("min lat: %.6f", s.minLat))
	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("max lat: %.6f", s.maxLat))
	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("min lon: %.6f", s.minLon))
	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("max lon: %.6f", s.maxLon))
	out.WriteString("\n")

	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("max depth (m): %.2f", s.maxDepth))
	out.WriteString("\n")

	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("west-east distance (m): %v", s.distanceWestEast))
	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("south-north distance (m): %v", s.distanceSouthNorth))
	out.WriteString("\n")

	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("grid lat (deg): %.10f", s.gridLat))
	out.WriteString("\n")

	out.WriteString(fmt.Sprintf("grid lon (deg): %.10f", s.gridLon))
	out.WriteString("\n")

	return out.String()
}

func (s *Stats) ParseLine(line string) {
	s.linenum++

	stringReader := strings.NewReader(line)

	csvReader := csv.NewReader(stringReader)
	csvReader.Comma = ','
	columns, err := csvReader.Read()
	if err != nil {
		log.Fatal(err)
	}

	lat, err := strconv.ParseFloat(columns[0], 64)
	if err == nil {
		s.minLat = math.Min(s.minLat, lat)
		s.maxLat = math.Max(s.maxLat, lat)
	} else {
		s.lineErrors = append(s.lineErrors, err)
	}

	lon, err := strconv.ParseFloat(columns[1], 64)
	if err == nil {
		s.minLon = math.Min(s.minLon, lon)
		s.maxLon = math.Max(s.maxLon, lon)
	} else {
		s.lineErrors = append(s.lineErrors, err)
	}

	depth, _ := strconv.ParseFloat(columns[2], 64)
	if err == nil {
		s.maxDepth = math.Max(s.maxDepth, depth)
	} else {
		s.lineErrors = append(s.lineErrors, err)
	}
}

func (s *Stats) ParseData(line string) {
	stringReader := strings.NewReader(line)

	csvReader := csv.NewReader(stringReader)
	csvReader.Comma = ','
	columns, _ := csvReader.Read()

	lat, _ := strconv.ParseFloat(columns[0], 64)
	lon, _ := strconv.ParseFloat(columns[1], 64)
	depth, _ := strconv.ParseFloat(columns[2], 64)

	lat = Round(lat, s.gridLat)
	lat = lat - s.minLat

	lon = Round(lon, s.gridLon)
	lon = lon - s.minLon

	depth = -1 * depth / s.maxDepth

	c := Coordinate{lat, lon}

	d, found := s.data[c]
	if found {
		s.data[c] = (d + depth) / 2
	} else {
		s.data[c] = depth
	}
}

func (s *Stats) Write(filename string) error {

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.WriteString(JS_HEAD)
	if err != nil {
		return err
	}

	for k, v := range s.data {
		_, err := file.WriteString(fmt.Sprintf("[%.6f, %.6f, %.2f],", k.lat, k.lon, v) + "\n")
		if err != nil {
			return err
		}
	}

	_, err = file.WriteString(JS_TAIL)
	if err != nil {
		return err
	}

	return nil

}

func (s *Stats) CheckError() *Stats {
	if len(s.lineErrors) > 0 {
		log.Fatal(s.lineErrors)
	}
	return s
}

func (s *Stats) Calc() *Stats {

	// 南北の距離
	southNorth := 2 * math.Pi * R * 1000 * (s.maxLat - s.minLat) / 360
	s.distanceSouthNorth = int(math.Floor(southNorth))

	// 東西の距離
	r := R * 1000 * math.Cos(degree2radian(s.minLat))
	westEast := 2 * math.Pi * r * (s.maxLon - s.minLon) / 360
	s.distanceWestEast = int(math.Floor(westEast))

	// グリッド単位
	s.gridLon = (s.maxLon - s.minLon) / (westEast / gridUnit)
	s.gridLat = (s.maxLat - s.minLat) / (southNorth / gridUnit)

	return s
}

func main() {
	filename := "./data/bathymetry_data.csv"

	// create Stats instance
	stats := NewStats()

	// read the file line by line
	err := ReadFile(filename, stats.ParseLine)
	if err != nil {
		log.Fatal(err)
	}

	stats.CheckError().Calc().Print()

	// again read the file and create Datas
	err = ReadFile(filename, stats.ParseData)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(len(stats.data))

	stats.Write("./data/data.js")

}
