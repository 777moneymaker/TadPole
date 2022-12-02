use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::{BufRead, BufReader, Lines};
use std::path::Path;

use anyhow::Result;
use nohash_hasher::NoHashHasher;

pub type PondMap = HashMap<u32, u32, BuildHasherDefault<NoHashHasher<u32>>>;

pub fn read_lines(filename: &Path) -> Result<Lines<BufReader<File>>> {
    let file = File::open(filename)?;
    Ok(BufReader::new(file).lines())
}

/// Builds a hashmap where key: prot\_id & value: hmm\_id:
/// hashmap used doesn't hash actually because its not needed here
/// making it faster;
/// # Examples
/// ```rust
/// use poteznyparser::*;
/// use std::collections::HashMap;
/// use std::hash::BuildHasherDefault;
///
/// let mut pond: PondMap = PondMap::default();
/// let some_lines = r#"AB036666|3,phrog_239,9,519,2.1e-51,180.4
///                     AB036666|29,phrog_11806,1,141,7.7e-59,204.2"#;
/// let phrog_lines = some_lines.split('\n').map(|x| x.to_string());
/// extract_id_phrog_relationship(&mut pond, phrog_lines).unwrap();
///
/// assert_eq!(pond[&3], 239);
/// assert_eq!(pond[&29], 11806);
/// ```
pub fn extract_id_phrog_relationship<L>(pond: &mut PondMap, phrog_lines: L) -> Result<()>
where
    L: Iterator<Item = String>,
{
    for line in phrog_lines {
        // AB036666|2,phrog_2199,16,407,5.5e-42,149.4
        let (_, rest) = line.split_once('|').expect("where |?"); // 2,phrog_2199,16,407,5.5e-42,149.4
        let (id, rest) = rest.split_once(',').unwrap(); // 2 & phrog_2199,16,407,5.5e-42,149.4
        let id = id.parse::<u32>()?;
        let (phrog, _) = rest.split_once(',').unwrap(); // phrog_2199
        let (_, phrog_number) = phrog.split_once('_').unwrap(); // 2199
        let phrog_number = phrog_number.parse::<u32>()?;
        pond.insert(id, phrog_number);
    }
    Ok(())
}

/// turns phrog_file into iterator over lines, skips the header
/// and passes it along to extract_id_phrog_relationship
/// for some real action
pub fn parse_phrog_file(pond: &mut PondMap, phrog_file: &Path) -> Result<()> {
    pond.clear();
    let phrog_lines = read_lines(phrog_file)?.skip(1).map(|x| x.unwrap());
    extract_id_phrog_relationship(pond, phrog_lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHROG_CSV: &str = r#"prot_id,hmm_id,prot_start,prot_end,evalue,score
    AB036666|2,phrog_2199,16,407,5.5e-42,149.4
    AB036666|3,phrog_239,9,519,2.1e-51,180.4
    AB036666|5,phrog_25990,1,367,1e-148,501.7
    AB036666|6,phrog_21102,1,409,3.4e-173,582.6
    AB036666|7,phrog_18320,39,150,1.9e-39,140.6
    AB036666|8,phrog_498,3,398,3.4e-144,488.0
    AB036666|9,phrog_57,1,158,2.5e-30,112.0
    AB036666|10,phrog_9049,1,484,1.3e-131,445.6
    AB036666|11,phrog_15,4,592,1.1e-259,871.5
    AB036666|12,phrog_75,1,65,3.1e-13,55.5
    AB036666|13,phrog_22762,7,255,4e-81,279.0
    AB036666|14,phrog_22762,1,74,1.5e-19,76.7
    AB036666|15,phrog_21,12,465,5.4e-116,394.9
    AB036666|16,phrog_53,30,249,3.1e-89,305.9
    AB036666|17,phrog_49,1,122,2.9e-26,98.3
    AB036666|18,phrog_29,1,331,3e-76,263.6
    AB036666|19,phrog_7787,17,113,3.5e-31,113.3
    AB036666|20,phrog_27,10,116,1.4e-18,73.8
    AB036666|21,phrog_1072,7,153,8.2e-48,168.1
    AB036666|22,phrog_9992,8,148,2.9e-31,114.9
    AB036666|23,phrog_281,10,78,8.5e-17,66.8
    AB036666|24,phrog_43,1,103,4.4e-38,135.9
    AB036666|25,phrog_4226,62,263,6.6e-45,158.8
    AB036666|26,phrog_17612,1,385,3.3e-214,717.6
    AB036666|27,phrog_9262,59,198,5.7e-86,226.3
    AB036666|28,phrog_9262,2,155,3.6e-64,222.9
    AB036666|29,phrog_11806,1,141,7.7e-59,204.2
    AB036666|30,phrog_95,3,452,3.7e-106,359.7
    AB036666|32,phrog_16592,4,642,2.4e-263,876.9"#;

    #[test]
    fn exctractor_test() {
        let mut pond: PondMap =
            HashMap::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let phrog_lines = PHROG_CSV.split('\n').skip(1).map(|x| x.to_string());
        extract_id_phrog_relationship(&mut pond, phrog_lines).unwrap();
        assert_eq!(pond[&2], 2199);
        assert_eq!(pond[&11], 15);
        assert_eq!(pond[&29], 11806);
        assert_eq!(pond[&32], 16592);
    }
}
