use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;

use poteznyparser::{parse_phrog_file, read_lines, PondMap};

const DEFAULT_OUTPUT_NAME: &str = "potezny_output.txt";
const UNKNOWN: &str = "joker";

fn get_files(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = dir_path.read_dir()?.map(|x| x.unwrap().path()).collect();
    files.sort();
    Ok(files)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to dir where one could find some phrogs
    phrog_dir: PathBuf,

    /// path to dir where one could find some gffs
    gff_dir: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    /// distance above which poteznyparser should start new sentence
    #[arg(short, long, default_value_t = 1000)]
    distance: usize,

    /// collapse jokers into Njoker
    #[arg(short, long, default_value_t = false)]
    collapse: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strand {
    Pos,
    Neg,
}

impl From<&str> for Strand {
    fn from(strand: &str) -> Self {
        match &strand[0..1] {
            "+" => Self::Pos,
            "-" => Self::Neg,
            _ => unreachable!("oof this is neither a + nor a -"),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ParsedLine {
    id: u32,
    phrog: Option<u32>,
    start: u32,
    end: u32,
    strand: Strand,
}

impl ParsedLine {
    fn from_gff_line(line: &str, pond: &PondMap) -> Result<Self> {
        let mut pos = line.split('\t');
        let start = pos.nth(3).unwrap().parse()?;
        let end = pos.next().unwrap().parse()?;
        let strand = Strand::from(pos.nth(1).unwrap());
        let (_, id) = pos.nth(1).unwrap().split_once('|').unwrap();
        let (id, _) = id.split_once(';').unwrap();
        let id: u32 = id.parse()?;
        let phrog = pond.get(&id).copied();

        Ok(ParsedLine {
            id,
            phrog,
            start,
            end,
            strand,
        })
    }

    fn gib_phrog_as_string(&self) -> String {
        match self.phrog {
            Some(phrog) => phrog.to_string(),
            None => UNKNOWN.to_string(),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let phrog_files = get_files(&args.phrog_dir)?;
    let gff_files = get_files(&args.gff_dir)?;
    // todo: what is the usual size of phrog csvs?
    // assign better capacity based on this knowledge
    let mut pond: PondMap = HashMap::with_capacity_and_hasher(100, BuildHasherDefault::default());

    let output_path = args.output.unwrap_or_else(|| DEFAULT_OUTPUT_NAME.into());
    let outfile = File::create(&output_path).expect("Unable to create file");
    let mut outfile = BufWriter::new(outfile);
    let mut sentence: Vec<String> = Vec::with_capacity(40);
    
    let file_num = phrog_files.len();
    let mut file_count = 1;

    for (phrog_file, gff_file) in phrog_files.iter().zip(gff_files.iter()) {
        parse_phrog_file(&mut pond, phrog_file)?; // after this pond is loaded with prot_id: hmm_id

        let mut parsed_lines = read_lines(gff_file)?
            .map(|x| x.unwrap())
            .skip_while(|x| x.starts_with('#'))
            .map(|gffl| ParsedLine::from_gff_line(&gffl, &pond));

        let mut previous = match parsed_lines.next() {
            Some(v) => v?,
            None => {
                eprintln!("{:?} oofed? is it empty", &gff_file);
                continue;
            }
        };

        sentence.push(previous.gib_phrog_as_string());
        for parsed_line in &mut parsed_lines.map(|x| x.expect("couldn't parse gff line")) {
            match (previous.strand, parsed_line.strand) {
                (Strand::Neg, Strand::Pos) => {
                    sentence.reverse();
                    writeln!(&mut outfile, "{}", sentence.join(" "))?;
                    sentence.clear()
                }
                (Strand::Pos, Strand::Neg) => {
                    writeln!(&mut outfile, "{}", sentence.join(" "))?;
                    sentence.clear()
                }
                _ => (),
            };
            sentence.push(parsed_line.gib_phrog_as_string());
            previous = parsed_line;
        }

        if previous.strand == Strand::Neg {
            sentence.reverse();
        }
        writeln!(&mut outfile, "{}", sentence.join(" "))?;
        print!("\rDone {}/{} files...", file_count, file_num);

        file_count += 1;
    }

    println!("yay! created: {:?}", &output_path);
    Ok(())
}
