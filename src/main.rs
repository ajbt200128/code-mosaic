use std::{collections::BTreeMap, fs};

use deltae::{DEMethod::DE2000, DeltaE, LabValue};
use image::{
    imageops::{overlay, FilterType::Nearest},
    io::Reader,
    DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgba, RgbaImage,
};
use lab::Lab;
use silicon::{
    directories::PROJECT_DIRS,
    formatter::{ImageFormatter, ImageFormatterBuilder},
    utils::{init_syntect, read_from_bat_cache},
};
use syntect::{
    dumps,
    easy::HighlightLines,
    highlighting::{Theme, ThemeSet},
    parsing::SyntaxSet,
    util::LinesWithEndings,
};
use tree_sitter::{Parser, Query, QueryCursor};

#[derive(Copy, Clone, Debug)]
enum Language {
    Rust,
    C,
}

const SUB_SIZE: u32 = 400;
const TILE_NUM: u32 = 80;
type ThemeMapping = (Rgba<u8>,Theme);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let original_image = Reader::open("mona.jpg")?.decode()?;
    let small = original_image.resize(TILE_NUM, TILE_NUM, image::imageops::FilterType::Gaussian);
    small.save("test.jpg").unwrap();
    let (ps, mut ts) = init_syntect();
    ts.add_from_folder("./assets/themes").unwrap();
    let funcs_rust = get_funcs(Language::Rust, "rust.rs");
    let funcs_c = get_funcs(Language::C, "c.c");
    let mut label_funcs_rust: Vec<(String, Language)> = funcs_rust
        .into_iter()
        .map(|s| (s, Language::Rust))
        .collect();
    let mut label_funcs_c: Vec<(String, Language)> =
        funcs_c.into_iter().map(|s| (s, Language::C)).collect();
    label_funcs_rust.append(&mut label_funcs_c);
    //let mut funcs:Vec<(String,Language)> = label_funcs_rust.into_iter().filter(|(s,l)| s.as_bytes().iter().filter(|&&c| c == b'\n').count() >= 25).collect();
    let mut funcs:Vec<(String,Language)> = label_funcs_rust;
    funcs.reverse();
    let mut row_handles = vec![];
    let width = small.width();
    let mappings = produce_mapping(funcs.pop().unwrap().0, funcs.pop().unwrap().1, ts.themes.clone(), ps.clone());
    for x in 0..small.height() {
        let mut code_vec = funcs.split_off(funcs.len() - width as usize);
        let pixel_row: Vec<(u32, u32, Rgba<u8>)> = small
            .pixels()
            .collect::<Vec<(u32, u32, Rgba<u8>)>>()
            .to_vec();
        let mut pixel_row = pixel_row[(x * width) as usize..((x + 1) * width) as usize].to_vec();
        let themes = ts.themes.clone();
        let map = mappings.clone();
        let ps_t = ps.clone();
        let h = tokio::spawn(async move {
            let mut best_pics = vec![];
            for _ in 0..width {
                let color = pixel_row.pop().unwrap().2;
                let (code, lang) = code_vec.pop().unwrap();
                //let res = calculate_single(code, lang, color, themes.clone(), ps_t.clone()).await;
                let res = calculate_single_cheap(code, lang, color, map.clone(), ps_t.clone()).await;

                best_pics.push(res.resize_to_fill(SUB_SIZE, SUB_SIZE, Nearest));
            }
            best_pics
        });
        row_handles.push(h);
    }
    println!("Handles made!");
    let mut y_cntr = 0;
    let mut canvas: RgbaImage =
        ImageBuffer::new(small.width() * SUB_SIZE, small.height() * SUB_SIZE);
    for h in row_handles {
        let mut x_cntr = 0;
        for p in h.await.unwrap() {
            overlay(
                &mut canvas,
                &p,
                small.width() * SUB_SIZE - x_cntr * SUB_SIZE,
                y_cntr * SUB_SIZE,
            );
            drop(p);
            x_cntr += 1;
        }
        println!("Row {} done!", &y_cntr);
        y_cntr += 1;
    }
    canvas.save("output.png").unwrap();
    Ok(())
}

fn produce_mapping(
    code: String,
    language: Language,
    themes: BTreeMap<String, Theme>,
    ps: SyntaxSet,
)->Vec<ThemeMapping>{
    let mut formatter = new_formatter();
    let mut v = vec![];
    for theme in themes.iter(){
        let img = produce_image(&code, language, &mut formatter, theme.1, &ps);
        let p = img.resize_exact(1, 1, Nearest).get_pixel(0, 0);
        v.push((p,theme.1.clone()));
    }
    v
}

fn new_formatter() -> ImageFormatter {
    ImageFormatterBuilder::new()
        .line_pad(5)
        .window_controls(false)
        .line_number(true)
        .round_corner(false)
        .tab_width(1)
        .font(vec![("Source Code Pro", 15.0), ("Fira Code", 15.0)])
        .build()
        .unwrap()
}

async fn calculate_single(
    code: String,
    language: Language,
    color: Rgba<u8>,
    themes: BTreeMap<String, Theme>,
    ps: SyntaxSet,
) -> DynamicImage {
    let mut best_dist = None;
    let mut best = None;
    let mut formatter = new_formatter();
    let code = code.replace(|c: char| !c.is_ascii(), "");
    for theme in themes.iter() {
        let code_image = produce_image(&code, language, &mut formatter, &theme.1, &ps);
        let new_dist = get_distance_from_image(&code_image, &color);
        if let Some(dist) = best_dist {
            if dist > new_dist {
                best_dist = Some(new_dist);
                best = Some(code_image);
            }
        } else {
            best_dist = Some(new_dist);
            best = Some(code_image);
        }
    }
    best.unwrap()
}

async fn calculate_single_cheap(
    code: String,
    language: Language,
    color: Rgba<u8>,
    themes: Vec<ThemeMapping>,
    ps: SyntaxSet,
) -> DynamicImage {
    let mut best_dist = None;
    let mut best = None;
    let mut formatter = new_formatter();
    let code = code.replace(|c: char| !c.is_ascii(), "");
    for theme in themes.iter() {
        //let code_image = produce_image(&code, language, &mut formatter, &theme.1, &ps);
        let new_dist = get_distance(&theme.0, &color);
        if let Some(dist) = best_dist {
            if dist > new_dist {
                best_dist = Some(new_dist);
                best = Some(theme.1.clone());
            }
        } else {
            best_dist = Some(new_dist);
            best = Some(theme.1.clone());
        }
    }
    let best = best.unwrap();
    produce_image(&code, language, &mut formatter, &best, &ps)
}

fn get_distance_from_image(img: &DynamicImage, color_b: &Rgba<u8>) -> f32 {
    let one = img.resize_exact(1, 1, Nearest);
    let color_a = GenericImageView::get_pixel(&one, 0, 0);
    get_distance(&color_a, color_b)
}

fn get_distance(color_a:&Rgba<u8>, color_b: &Rgba<u8>) -> f32 {
    let color_a = Lab::from_rgb(&color_a.to_rgb().0);
    let color_b = Lab::from_rgb(&color_b.to_rgb().0);
    let color_a = LabValue {
        a: color_a.a,
        b: color_a.b,
        l: color_a.l,
    };
    let color_b = LabValue {
        a: color_b.a,
        b: color_b.b,
        l: color_b.l,
    };
    let delta = DeltaE::new(color_a, color_b, DE2000);
    *delta.value()
}

fn get_funcs(language: Language, path: &str) -> Vec<String> {
    let text = fs::read_to_string(path).expect("File does not exist");
    let mut parser = Parser::new();
    let parse_lang = match language {
        Language::Rust => tree_sitter_rust::language(),
        Language::C => tree_sitter_c::language(),
    };
    parser
        .set_language(parse_lang)
        .expect("Tree sitter did not load!");
    let tree = parser.parse(text.clone(), None).unwrap();
    let q_exp = match language {
        Language::Rust => "(function_item) @m",
        Language::C => "(function_definition) @m",
    };
    let query = Query::new(parse_lang, q_exp).unwrap();
    let mut cursor = QueryCursor::new();
    let captures = cursor.captures(&query, tree.root_node(), text.as_bytes());
    captures
        .into_iter()
        .map(|c| {
            c.0.captures[0]
                .node
                .utf8_text(text.as_bytes())
                .unwrap()
                .to_string()
        })
        .collect()
}

fn produce_image(
    text: &str,
    language: Language,
    formatter: &mut ImageFormatter,
    theme: &Theme,
    ps: &SyntaxSet,
) -> DynamicImage {
    let language = match language {
        Language::Rust => "Rust",
        Language::C => "C",
    };
    let language = ps
        .find_syntax_by_token(language)
        .expect("Unsuppported language");
    let mut h = HighlightLines::new(language, &theme);
    let highlight = LinesWithEndings::from(&text)
        .map(|line| h.highlight(line, &ps))
        .collect::<Vec<_>>();
    formatter.format(&highlight, &theme)
}
