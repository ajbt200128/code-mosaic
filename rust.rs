use std::io::{BufWriter, Cursor};

use image::{GenericImageView,GenericImage, ImageBuffer, EncodableLayout, Rgba, RgbaImage, Pixel, DynamicImage};
use imageproc::drawing::{draw_text, draw_text_mut};
use rusttype::{Font, Scale};
use serde_json::json;
use worker::*;

mod utils;

fn log_request(req: &Request) {
    console_log!(
        "{} - [{}], located at: {:?}, within: {}",
        Date::now().to_string(),
        req.path(),
        req.cf().coordinates().unwrap_or_default(),
        req.cf().region().unwrap_or("unknown region".into())
    );
}

const TEXT_COLOR: image::Rgba<u8> = image::Rgba([255,255,255,255]);

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: worker::Context) -> Result<Response> {
    log_request(&req);
    // Optionally, get more helpful error messages written to the console in the case of a panic.
    utils::set_panic_hook();

    // Optionally, use the Router to handle matching endpoints, use ":name" placeholders, or "*name"
    // catch-alls to match on specific patterns. Alternatively, use `Router::with_data(D)` to
    // provide arbitrary data that will be accessible in each route via the `ctx.data()` method.
    let router = Router::new();
    // Add as many routes as your Worker needs! Each route will get a `Request` for handling HTTP
    // functionality and a `RouteContext` which you can use to  and get route parameters and
    // Environment bindings like KV Stores, Durable Objects, Secrets, and Variables.
    router
        .get("/", |_, _| Response::ok("Hello from Workers!"))
        .get("/worker-version", |_, ctx| {
            let version = ctx.var("WORKERS_RS_VERSION")?.to_string();
            Response::ok(version)
        })
        .get_async("/nft.jpg",|_,ctx| async move {
            let font_data: &[u8] = include_bytes!("/usr/share/fonts/FuturaLT-Bold.ttf");
            let font: Font<'static> = Font::try_from_bytes(font_data).unwrap();
            let kv =ctx.kv("CNTR")?;
            let mut cntr = if let Some(v) = kv.get("cntr").text().await?{
                v.parse().unwrap()
            }else{
                0
            };
            cntr+=1;
            let cntr_str = format!("{}",cntr);
            let cntr = kv.put("cntr", cntr)?.execute();
            let img:RgbaImage= ImageBuffer::new(256,256);
            let mut img = DynamicImage::ImageRgba8(img);
            draw_text_mut(&mut img, TEXT_COLOR,0,0, Scale::uniform(20.0), &font,&cntr_str );

            let mut bytes: Vec<u8> = Vec::new();
            img.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Jpeg(100)).unwrap();
            cntr.await?;
            Response::from_bytes(bytes)
        })
        .run(req, env)
        .await
}
#![feature(build_hasher_simple_hash_one)]
use std::cmp;
use std::hash::BuildHasher;
use std::u32;

use bloom::{BloomFilter, ASMS};
use hyperloglogplus::HyperLogLog;
//use hyperloglog::HyperLogLog;
use hyperloglogplus::HyperLogLogPF;
use rand::prelude::IteratorRandom;
use rand::Rng;
use std::collections::hash_map::RandomState;

/*
trait LogLog {
    fn insert(&mut self, value: u32);
    fn cardinality(&self) -> f64;
}

struct NormalLogLog {
    log: HyperLogLog<u32>,
}

impl NormalLogLog {
    fn new(error_rate: f32) -> Self {
        NormalLogLog {
            log: HyperLogLog::new(error_rate as f64),
        }
    }
}

impl LogLog for NormalLogLog {
    fn insert(&mut self, value: u32) {
        self.log.insert(&value)
    }

    fn cardinality(&self) -> f64 {
        self.log.len()
    }
}

struct HashLogLog {
    log: HyperLogLog<u32>,
    bloom: BloomFilter,
    prev_count: f64,
    error_rate: f32,
}

impl HashLogLog {
    fn new(error_rate: f32, expected_items: u32) -> Self {
        HashLogLog {
            log: HyperLogLog::new(error_rate as f64),
            bloom: BloomFilter::with_rate(error_rate, expected_items),
            prev_count: 0.0,
            error_rate,
        }
    }
    fn change_hash(&mut self) {
        self.prev_count += self.log.len();
        self.log = HyperLogLog::new(self.error_rate as f64);
    }
}

impl LogLog for HashLogLog {
    fn insert(&mut self, value: u32) {
        if !self.bloom.contains(&value) {
            self.log.insert(&value);
            self.bloom.insert(&value);
        }
    }

    fn cardinality(&self) -> f64 {
        self.log.len() + self.prev_count
    }
}
*/
fn sigma_round(est: f64, sigma: f64,l:i32) -> (f64,i32) {
    // find l st we minimize est/(1+sigma)^l,(1+sigma)^l/est;
    let mut curr_l = l;
    let mut last_est = (1.0 + sigma).powi(curr_l);
    let mut last_min = (last_est / est).max(est / last_est);
    curr_l += 1;
    let mut curr_est = (1.0 + sigma).powi(curr_l);
    let mut curr_min = (curr_est / est).max(est / curr_est);
    while last_min > curr_min {
        last_min = curr_min;
        last_est = curr_est;
        curr_l += 1;
        curr_est = (1.0 + sigma).powi(curr_l);
        curr_min = (curr_est / est).max(est / curr_est)
    }
    (last_est,curr_l)
}
fn main() {
    /*
    let mut l1_cnt = 0.0;
    let mut l2_cnt = 0.0;
    for _ in 1..10u16 {
        let mut rng = rand::thread_rng();
        let mut log1 = NormalLogLog::new(0.05);
        let mut log3 = NormalLogLog::new(0.05);
        let mut log2 = HashLogLog::new(0.05, 2_000_000);
        let mut list = vec![];
        for _ in 1..1_000_000u32 {
            let i = rng.gen_range(0..5_000_000);
            log1.insert(i);
            log2.insert(i);
            if i % 10000 == 0{
                log2.change_hash();
            }
            list.push(i);
        }
        list.sort_unstable();
        list.dedup();
        l1_cnt += ((list.len() as f64 - log1.cardinality()) /list.len() as f64).powf(2.0);
        l2_cnt += ((list.len() as f64 - log2.cardinality()) /list.len() as f64).powf(2.0);

       println!("Log 1 {}", log1.cardinality());
       println!("Log 2 {}", log2.cardinality());
       println!("Real {}",list.len());
        println!("m: {}",log1.log.len());
        println!("m: {}",log3.log.len());
    }

    let result1 = (l1_cnt/10.0).sqrt();
    let result2 = (l2_cnt/10.0).sqrt();
    println!("1: {} 2: {}",result1,result2);
     */
    //let est = 10.0;
    //let sigma = 0.05;
    //sigma_round(est, sigma);
    let mut v = vec![];
    let mut v_flip = vec![];
    let mut v_last_est = vec![];
    let mut v_l = vec![];
    let e = 12;
    let sigma = (1.0/e as f64)*4.0;
    let rs = RandomState::new();
    let l = HyperLogLogPF::<u32, _>::new(e, rs.clone()).unwrap();
    v.push(l);
    v_flip.push(0);
    v_last_est.push(0.0);
    v_l.push(0);
    for _ in 1..10 {
        let l = HyperLogLogPF::<u32, _>::new(e, RandomState::new()).unwrap();
        v.push(l);
        v_flip.push(0);
        v_last_est.push(0.0);
        v_l.push(0);
    }
    let mut rng = rand::thread_rng();
    /*
    let mut bigs = vec![];
    for _ in 0..15 {
        let mut max = 0;
        let mut max_h = rs.hash_one(max);
        for _ in 0..1_000_000 {
            let i = rng.gen_range(0..500_000_000);
            let i_h = rs.hash_one(i);
            if i_h.leading_zeros() > max_h.leading_zeros() && !bigs.contains(&i){
                max = i;
                max_h = i_h;
            }
        }
        bigs.push(max);
    }
    for max in bigs{
        println!("max:{} zeros: {}", max, rs.hash_one(max).leading_zeros());
        println!("est: {}", v[0].count());
        v[0].insert(&max);
        println!("est: {}", v[0].count());
    }
*/
    for _ in 0..1_000_000 {
        let i = rng.gen_range(0..5_000_000);
        v.iter_mut().for_each(|l| l.insert(&i));
    }
    for curr in 0..5_000_000 {
        let i = rng.gen_range(0..5_000_000);
        v.iter_mut().for_each(|l| l.insert(&i));
        //v.iter().for_each(|l| sigma_round(l.len(), sigma));
        let mut t_flip = 0;
        for i in 0..v_last_est.len() {
            if v[i].count() * (1.0 - sigma) >= v_last_est[i]
                || v[i].count() * (1.0 + sigma) <= v_last_est[i]
            {
                let (e,l) = sigma_round(v[i].count(), sigma,v_l[i]);
                v_last_est[i] = e;
                v_flip[i] += 1;
                t_flip += 1;
                v_l[i] = l;
            }
        }
        let mut not_in = vec![false; v.len()];
        for i in 0..v.len() {
            for j in 0..v_last_est.len() {
                if i != j
                    && (v[i].count() * (1.0 - sigma) >= v_last_est[j]
                        || v[i].count() * (1.0 + sigma) <= v_last_est[j])
                {
                    not_in[i] = true;
                }
            }
        }
        if t_flip > 0 {
            let range = v_flip.iter().max().unwrap() - v_flip.iter().min().unwrap();
            if range >0 {
                println!("step: {}", curr);
                v_flip.iter().for_each(|f| println!("flip: {}", f));
                println!("Total changed: {}", t_flip);
                println!("Range: {}", range);
                v_last_est.iter().for_each(|e| println!("sig est: {}", e));
                v.iter_mut().for_each(|e| println!("est: {}", e.count()));
                not_in.iter().for_each(|a| println!("{}", a));
                println!("------------------");
            }
        }
        // V X X
        // 0 0 0
    }
}
pub mod connector;
pub mod connection;
pub mod models;
#[cfg(test)]
pub mod tests;
use cfg_if::cfg_if;

cfg_if! {
    // https://github.com/rustwasm/console_error_panic_hook#readme
    if #[cfg(feature = "console_error_panic_hook")] {
        extern crate console_error_panic_hook;
        pub use self::console_error_panic_hook::set_once as set_panic_hook;
    } else {
        #[inline]
        pub fn set_panic_hook() {}
    }
}
use std::{error::Error, fs::File, io::{self, BufRead, Read, Write}, path::Path};

use clap::{Arg,App};

mod token;

static mut HAD_ERROR:bool = false;

fn main() {
    let matches = App::new("rlox")
        .version("1.0")
        .arg(Arg::with_name("source")
             .value_name("FILE")
             ).get_matches();
    if let Some(input) = matches.value_of("FILE"){
        run_file(Path::new(input));
    }else{
        run_prompt();
    }
}

fn run_file(path:&Path){
    let mut file = match File::open(path){
        Err(e) => panic!("Couldn't read source file {}: {}",path.to_str().unwrap(),e),
        Ok(file) => file
    };
    let mut s = String::new();
    file.read_to_string(&mut s).unwrap();
    run(s);
    unsafe{
        if HAD_ERROR {panic!("Error")};
    }
}

fn run(src: String){
    for t in src.split_whitespace(){

    }
}

fn run_prompt(){
    loop {
        let mut input = String::new();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut input).unwrap();
        run(input);
        unsafe{HAD_ERROR = false}
    }
}

fn error(line:u64, e:impl Error){
    eprintln!("{}: {}",line,e);
}

fn report(line:u64,loc:String, e:impl Error){
    eprintln!("[line {}] Error {}: {}",line,loc,e);
}
use std::{env, fs};
use std::path::PathBuf;

fn main() {
    // Put the memory definitions somewhere the linker can find it
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search={}", out_dir.display());

    let boards: Vec<_> = env::vars().filter_map(|(key, _value)| {
        if key.starts_with("CARGO_FEATURE_BOARD") {
            Some(key[20..].to_ascii_lowercase())  // Strip 'CARGO_FEATURE_BOARD_'
        } else {
            None
        }
    }).collect();

    if boards.is_empty() {
        panic!("No board features selected");
    }
    if boards.len() > 1 {
        panic!("More than one board feature selected: {:?}", boards);
    }

    let board = boards.first().unwrap();

    match board.as_str() {
        "hifive1" => {
            fs::copy("memory-hifive1.x", out_dir.join("hifive1-memory.x")).unwrap();
            println!("cargo:rerun-if-changed=memory-hifive1.x");
        }
        "hifive1_revb" => {
            fs::copy("memory-hifive1-revb.x", out_dir.join("hifive1-memory.x")).unwrap();
            println!("cargo:rerun-if-changed=memory-hifive1-revb.x");
        }
        "lofive" | "lofive_r1" => {
            fs::copy("memory-lofive-r1.x", out_dir.join("hifive1-memory.x")).unwrap();
            println!("cargo:rerun-if-changed=memory-lofive-r1.x");
        }

        other => panic!("Unknown board: {}", other),
    }

    fs::copy("hifive1-link.x", out_dir.join("hifive1-link.x")).unwrap();

    // Copy library with flash setup code
    let name = env::var("CARGO_PKG_NAME").unwrap();
    fs::copy("bin/flash.a", out_dir.join(format!("lib{}.a", name))).unwrap();
    println!("cargo:rustc-link-lib=static={}", name);
    println!("cargo:rerun-if-changed=bin/flash.a");
}
#![no_std]
#![no_main]

//extern crate panic_halt;

use riscv_rt::entry;
use hifive1::hal::prelude::*;
use hifive1::hal::spi::{Spi, MODE_0, SpiX};
use hifive1::hal::gpio::{gpio0::Pin10, Input, Floating};
use hifive1::hal::delay::Delay;
use hifive1::hal::clock::Clocks;
use hifive1::hal::DeviceResources;
use hifive1::{sprintln, pin};
use core::panic::PanicInfo;
use embedded_hal::blocking::delay::DelayUs;
use embedded_hal::blocking::spi::WriteIter;

#[inline(never)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    sprintln!("panic: {}", info);
    loop {
        use core::sync::atomic;
        use core::sync::atomic::Ordering;
        atomic::compiler_fence(Ordering::SeqCst);
    }
}

#[derive(Debug)]
enum EspError {
    ProtocolError,
    BufferOverflow,
    WouldBlock
}

struct EspWiFi<SPI, PINS> {
    spi: Spi<SPI, PINS>,
    handshake: Pin10<Input<Floating>>,
    delay: FastDelay,
}

impl<SPI: SpiX, PINS> EspWiFi<SPI, PINS> {
    fn send_bytes(&mut self, bytes: &[u8]) {
        self.delay.delay_us(18u32);
        self.spi.write(bytes).unwrap();
        self.delay.delay_us(5000u32);
    }

    fn transfer(&mut self, buffer: &mut [u8]) {
        self.delay.delay_us(18u32);
        self.spi.transfer(buffer).unwrap();
        self.delay.delay_us(5000u32);
    }

    fn discard(&mut self, size: usize) {
        self.delay.delay_us(18u32);
        self.spi.write_iter((0..size).map(|_| 0x00)).unwrap();
        self.delay.delay_us(5000u32);
    }

    pub fn send(&mut self, s: &str) {
        let bytes = s.as_bytes();
        assert!(bytes.len() <= 127);

        self.send_bytes(&[0x02, 0x00, 0x00, 0x00]);
        self.send_bytes(&[bytes.len() as u8, 0x00, 0x00, 0x41]);
        self.send_bytes(bytes);
    }

    pub fn recv<'a>(&mut self, buffer: &'a mut [u8]) -> Result<&'a str, EspError> {
        if self.handshake.is_low().unwrap() {
            return Err(EspError::WouldBlock);
        }

        self.send_bytes(&[0x01, 0x00, 0x00, 0x00]);

        let mut request = [0u8; 4];
        self.transfer(&mut request);
        if request[3] != 0x42 {
            return Err(EspError::ProtocolError);
        }

        let n = (request[0] & 0x7F) as usize + ((request[1] as usize) << 7);
        if n > buffer.len() {
            self.discard(n);
            return Err(EspError::BufferOverflow);
        }

        self.transfer(&mut buffer[..n]);
        Ok(core::str::from_utf8(&buffer[..n]).unwrap())
    }
}

struct FastDelay {
    us_cycles: u64,
}

impl FastDelay {
    pub fn new(clocks: Clocks) -> Self {
        Self {
            us_cycles: clocks.coreclk().0 as u64 * 3 / 2_000_000,
        }
    }
}

impl DelayUs<u32> for FastDelay {
    fn delay_us(&mut self, us: u32) {
        use riscv::register::mcycle;

        let t = mcycle::read64() + self.us_cycles * (us as u64);
        while mcycle::read64() < t {}
    }
}

#[entry]
fn main() -> ! {
    let dr = DeviceResources::take().unwrap();
    let p = dr.peripherals;
    let gpio = dr.pins;

    // Configure clocks
    let clocks = hifive1::clock::configure(p.PRCI, p.AONCLK, 320.mhz().into());

    // Configure UART for stdout
    hifive1::stdout::configure(p.UART0, pin!(gpio, uart0_tx), pin!(gpio, uart0_rx), 115_200.bps(), clocks);

    // Configure SPI pins
    let mosi = pin!(gpio, spi0_mosi).into_iof0();
    let miso = pin!(gpio, spi0_miso).into_iof0();
    let sck = pin!(gpio, spi0_sck).into_iof0();
    let cs = pin!(gpio, spi0_ss2).into_iof0();

    // Configure SPI
    let pins = (mosi, miso, sck, cs);
    let spi = Spi::new(p.QSPI1, pins, MODE_0, 100_000.hz(), clocks);
    let mem = Spi::new(p.QSPI0, (), MODE_0, 100_000.hz(), clocks);

    let handshake = gpio.pin10.into_floating_input();
    let mut wifi = EspWiFi {
        spi,
        handshake,
        delay: FastDelay::new(clocks),
    };

    sprintln!("WiFi Test !!");

    Delay.delay_ms(10u32);

    let mut buffer = [0u8; 256];

    wifi.send("AT+CWMODE=2\r\n");
    Delay.delay_ms(20u32);
    sprintln!("resp: {:?}", wifi.recv(&mut buffer));
  //Delay.delay_ms(20u32);
  //wifi.send("AT+CWJAP=\"The Theriaults\",\"Buddydog1@\" \r\n");
  //sprintln!("resp: {:?}", wifi.recv(&mut buffer));

    loop {
    }
}
use arrayfire::{homography, print, Array, Dim4};
use eframe::{
    egui::{
        self,
        plot::{Plot, PlotImage, Value},
    },
    epaint::ColorImage,
    epi,
};
use egui_extras::RetainedImage;
use image::{
    imageops::resize, io::Reader, DynamicImage, GenericImage, GenericImageView, ImageBuffer, Pixel,
    RgbaImage,
};
use imageproc::geometric_transformations::{warp_into, Projection};

pub struct MosaicApp {
    image_a: RetainedImage,
    image_b: RetainedImage,
    image_a_orig: DynamicImage,
    image_b_orig: DynamicImage,
    points_a: Vec<Value>,
    points_b: Vec<Value>,
    warped: Option<RetainedImage>,
    warped_orig: Option<DynamicImage>,
}

impl Default for MosaicApp {
    fn default() -> Self {
        let im1 = Reader::open("imgs/a.jpg").unwrap().decode().unwrap();
        let im2 = Reader::open("imgs/b.jpg").unwrap().decode().unwrap();
        Self {
            image_a: to_retained("image_a", im1.clone()),
            image_b: to_retained("image_b", im2.clone()),
            image_a_orig: im1,
            image_b_orig: im2,
            points_a: vec![],
            points_b: vec![],
            warped: None,
            warped_orig: None,
        }
    }
}

fn to_retained(debug_name: impl Into<String>, im: DynamicImage) -> RetainedImage {
    let size = [im.width() as _, im.height() as _];
    let mut pixels = im.to_rgba8();
    let pixels = pixels.as_flat_samples_mut();
    RetainedImage::from_color_image(
        debug_name,
        ColorImage::from_rgba_unmultiplied(size, pixels.as_slice()),
    )
}

fn clamp_add(a: u8, b: u8, max: u8) -> u8 {
    if (a as u16 + b as u16) > max.into() {
        max
    } else {
        a + b
    }
}

fn distance_alpha((a_x, a_y): (f64, f64), (b_x, b_y): (f64, f64), max: u32) -> u8 {
    255 - (((a_x - b_x).powf(2.0) + (a_y - b_y).powf(2.0)).sqrt() / max as f64) as u8
}

fn overlay_into(a: &DynamicImage, b: &mut DynamicImage, center: (f64, f64)) {
    let mut b_n = b.clone();
    println!("a-------");
    for y in 0..a.height() {
        for x in 0..a.width() {
            let mut p = a.get_pixel(x, y);
            let mut q = b.get_pixel(x, y);
            //p.0[3] = distance_alpha(center, (x as f64, y as f64), b.width());
            if p.0[3] == 0 {
                p = q;
            } else if q.0[3] != 0 {
                q.0[3] = 125;
                p.0[3] = 125;
                p.blend(&q);
            }
            b_n.put_pixel(x, y, p);
        }
    }
    b_n.save("dbg.jpg");
    let b_n_a: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n,
        b_n.width() / 8,
        b_n.height() / 8,
        image::imageops::FilterType::Nearest,
    ));
    println!("b-----");
    b_n_a.blur(100.0);
    let b_n_b: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n_a,
        b_n_a.width() / 8,
        b_n_a.height() / 8,
        image::imageops::FilterType::Nearest,
    ));
    b_n_b.blur(200.0);
    println!("b-----");
    let b_n_a: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n_a,
        b.width(),
        b.height(),
        image::imageops::FilterType::Nearest,
    ));
    let b_n_b: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n_b,
        b.width(),
        b.height(),
        image::imageops::FilterType::Nearest,
    ));
    println!("c------");
    for y in 0..b.height() {
        for x in 0..b.width() {
            let mut p = b_n.get_pixel(x, y);
            let mut p_a = b_n_a.get_pixel(x, y);
            let mut p_b = b_n_b.get_pixel(x, y);
            let mut q = b.get_pixel(x, y);

            let mut r = if x < a.width() && y < a.height() {
                a.get_pixel(x, y)
            } else {
                image::Rgba([0, 0, 0, 0])
            };
            //if r.0[3] == 0 && q.0[3] != 0{
            //    p = q
            //}else if r.0[3] != 0 && q.0[3] == 0{
            //    p = r;
            //}else{
            p_a.0[3] = 185;
            // Smallest
            p_b.0[3] = 125;

            // Blend all three photos together
            p_a.blend(&p_b);
None            p.blend(&p_a);
            // Set alpha according to distance from center
            p.0[3] = distance_alpha(center, (x as f64, y as f64), b.width());

            // Blend first photo and all merged photos
            if r.0[3] != 0 {
                r.0[3] = 150;
                p.blend(&r);
            }
            p.0[3] = 255;

            
            //}

            b.put_pixel(x, y, p);
        }
    }
    println!("d------");
}

fn find_homography(a: Vec<Value>, b: Vec<Value>) -> [f32; 9] {
    let mut v = [1.0; 9];
    let mut x_src = [0.0; 4];
    let mut y_src = [0.0; 4];
    let mut x_dst = [0.0; 4];
    let mut y_dst = [0.0; 4];
    for i in 0..a.len() {
        x_src[i] = a[i].x as f32;
        y_src[i] = a[i].y as f32;
        x_dst[i] = b[i].x as f32;
        y_dst[i] = b[i].y as f32;
    }
    let x_src = Array::new(&x_src, Dim4::new(&[4, 1, 1, 1]));
    let y_src = Array::new(&y_src, Dim4::new(&[4, 1, 1, 1]));
    let x_dst = Array::new(&x_dst, Dim4::new(&[4, 1, 1, 1]));
    let y_dst = Array::new(&y_dst, Dim4::new(&[4, 1, 1, 1]));
    let (h, i): (Array<f32>, i32) = homography(
        &x_src,
        &y_src,
        &x_dst,
        &y_dst,
        arrayfire::HomographyType::RANSAC,
        100000.0,
        10,
    );

    print(&h);
    h.host(&mut v);
    v
}

impl epi::App for MosaicApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let plot_image_a = PlotImage::new(
                self.image_a.texture_id(ctx),
                egui::plot::Value {
                    x: (self.image_a.size_vec2().x / 2.0) as f64,
                    y: (self.image_a.size_vec2().y / 2.0) as f64,
                },
                self.image_a.size_vec2(),
            );

            let plot_image_b = PlotImage::new(
                self.image_b.texture_id(ctx),
                egui::plot::Value {
                    x: (self.image_b.size_vec2().x / 2.0) as f64,
                    y: (self.image_b.size_vec2().y / 2.0) as f64,
                },
                self.image_b.size_vec2(),
            );
            let plot_a = Plot::new("image_a_plot");
            let plot_b = Plot::new("image_b_plot");
            let plot_c = Plot::new("image_c_plot");
            //let img_plot = PlotImage::new(texture_id, center_position, size)

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    plot_a
                        .allow_drag(false)
                        .show_axes([false, false])
                        .height(800.0)
                        .width(800.0)
                        .show(ui, |plot_ui| {
                            plot_ui.image(plot_image_a.name("image_a"));
                            if plot_ui.plot_clicked() {
                                let mut coord = plot_ui.pointer_coordinate().unwrap();
                                coord.y = self.image_a_orig.height() as f64 - coord.y;
                                self.points_a.insert(0, coord);
                                if self.points_a.len() > 4 {
                                    self.points_a.pop();
                                }
                            }
                        });
                    plot_b
                        .allow_drag(false)
                        .show_axes([false, false])
                        .height(800.0)
                        .width(800.0)
                        .show(ui, |plot_ui| {
                            plot_ui.image(plot_image_b.name("image_b"));
                            if plot_ui.plot_clicked() {
                                let mut coord = plot_ui.pointer_coordinate().unwrap();
                                coord.y = self.image_b_orig.height() as f64 - coord.y;
                                self.points_b.insert(0, coord);
                                if self.points_b.len() > 4 {
                                    self.points_b.pop();
                                }
                            }
                        });
                });
                if self.warped.is_some() {
                    if ui.button("save").clicked() {
                       self.warped_orig.clone().unwrap().save("out.jpg");

                    }
                    let plot_image_c = PlotImage::new(
                        self.warped.as_ref().unwrap().texture_id(ctx),
                        egui::plot::Value {
                            x: (self.image_b.size_vec2().x / 2.0) as f64,
                            y: (self.image_b.size_vec2().y / 2.0) as f64,
                        },
                        self.image_b.size_vec2(),
                    );
                    plot_c
                        .allow_drag(false)
                        .show_axes([false, false])
                        .height(800.0)
                        .width(1600.0)
                        .show(ui, |plot_ui| {
                            plot_ui.image(plot_image_c.name("image_c"));
                        });
                }
            });
            if ui.button("Merge").clicked() {
                if self.points_a.len() == 4 && self.points_b.len() == 4 {
                    let h = find_homography(self.points_b.clone(), self.points_a.clone());
                    let projection = Projection::from_matrix(h).unwrap();
                    let white: image::Rgba<u8> = image::Rgba([0, 0, 0, 0]);
                    let mut canvas: RgbaImage =
                        ImageBuffer::new(self.image_a_orig.width() * 2, self.image_a_orig.height());
                    warp_into(
                        &self.image_b_orig.to_rgba8(),
                        &projection,
                        imageproc::geometric_transformations::Interpolation::Nearest,
                        white,
                        &mut canvas,
                    );
                    let mut canvas = image::DynamicImage::ImageRgba8(canvas);
                    let x = self
                        .points_a
                        .clone()
                        .iter()
                        .fold(0.0, |cntr, curr| cntr + curr.x)
                        / 4.0;
                    let y = self
                        .points_a
                        .clone()
                        .iter()
                        .fold(0.0, |cntr, curr| cntr + curr.y)
                        / 4.0;
                    overlay_into(&self.image_a_orig, &mut canvas, (x, y));
                    self.warped_orig = Some(canvas.clone());
                    self.warped = Some(to_retained("w", canvas));
                    self.points_a = vec![];
                    self.points_b = vec![];
                }
            }
            egui::warn_if_debug_build(ui);
        });
    }
}
#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::MosaicApp;

// ----------------------------------------------------------------------------
// When compiling for web:

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), eframe::wasm_bindgen::JsValue> {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let app = MosaicApp::default();
    eframe::start_web(canvas_id, Box::new(app))
}
use std::{convert::TryInto, default, fmt::Debug, io::BufRead, iter::Peekable, str::Chars};

#[derive(Debug)]
#[repr(u8)]
enum TokenType {
    // Single-character tokens.
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,

    // oNE OR TWO CHARACTER TOKENS.
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    // lITERALS.
    Identifier(String),
    String(String),
    Number(f64),

    // kEYWORDS.
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    Eof,
    Unknown(char),
}
#[derive(Debug)]
struct Literal {}

#[derive(Debug)]
struct Token {
    t: TokenType,
    literal: Option<Literal>,
    line: u64,
}

struct Scanner<'a> {
    current_position: u32,
    it: Peekable<Chars<'a>>,
    end: bool,
    line: u64,
}

impl<'a> Scanner<'a> {
    fn new(source: &'a str) -> Self {
        Scanner {
            current_position: 0,
            it: source.chars().peekable(),
            end: false,
            line: 0,
        }
    }
}
impl Scanner<'_> {
    fn look_ahead(&mut self, c: char) -> bool {
        if self.it.peek().is_some() && self.it.peek().unwrap().to_owned() == c {
            self.it.next();
            true
        } else {
            false
        }
    }
    fn match_token(&mut self, c: char) -> Option<TokenType> {
        match c {
            '=' => Some(if self.look_ahead('=') {
                TokenType::EqualEqual
            } else {
                TokenType::Equal
            }),
            '!' => Some(if self.look_ahead('=') {
                TokenType::BangEqual
            } else {
                TokenType::Bang
            }),
            '<' => Some(if self.look_ahead('=') {
                TokenType::LessEqual
            } else {
                TokenType::Less
            }),
            '>' => Some(if self.look_ahead('=') {
                TokenType::GreaterEqual
            } else {
                TokenType::Greater
            }),
            ' ' => None,
            '/' => if self.look_ahead('/') {
                while self.it.peek() != None || self.it.peek() != Some(&'\n') {
                    self.it.next();
                }
                self.line+=1;
                None
            } else {
                Some(TokenType::Slash)
            },
            '"' => {
                let mut chars = vec![];
                
                while let Some(&n) = self.it.peek(){
                    if n == '"' {
                        break;
                    }
                    chars.push(n);
                }
                Some(TokenType::String(chars.into_iter().collect()))
            },
            '\n' => {self.line+=1; None},
            '\r' => None,
            '\t' => None,
            '.' => Some(TokenType::Dot),
            '(' => Some(TokenType::LeftParen),
            ')' => Some(TokenType::RightParen),
            '{' => Some(TokenType::LeftBrace),
            '}' => Some(TokenType::RightBrace),
            ',' => Some(TokenType::Comma),
            '-' => Some(TokenType::Minus),
            '+' => Some(TokenType::Plus),
            ';' => Some(TokenType::Semicolon),
            '*' => Some(TokenType::Star),
            _ => Some(TokenType::Unknown(c)),
        }
    }

    fn scan_token(&mut self, c: char) -> Option<Token> {
        if let Some(t) = self.match_token(c){
            Some(Token{t,literal:None,line:self.line})
        }else{
            None
        }
    }
}

impl Iterator for Scanner<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.it.next() {
            self.scan_token(c)
        } else if !self.end {
            self.end = true;
            Some(Token {
                t: TokenType::Eof,
                literal: None,
                line: self.line,
            })
        } else {
            None
        }
    }
}
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "perf-literal")]
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use syntax::hir::literal::Literals;
use syntax::hir::Hir;
use syntax::ParserBuilder;

use backtrack;
use cache::{Cached, CachedGuard};
use compile::Compiler;
#[cfg(feature = "perf-dfa")]
use dfa;
use error::Error;
use input::{ByteInput, CharInput};
use literal::LiteralSearcher;
use pikevm;
use prog::Program;
use re_builder::RegexOptions;
use re_bytes;
use re_set;
use re_trait::{Locations, RegularExpression, Slot};
use re_unicode;
use utf8::next_utf8;

/// `Exec` manages the execution of a regular expression.
///
/// In particular, this manages the various compiled forms of a single regular
/// expression and the choice of which matching engine to use to execute a
/// regular expression.
pub struct Exec {
    /// All read only state.
    ro: Arc<ExecReadOnly>,
    /// Caches for the various matching engines.
    cache: Cached<ProgramCache>,
}

/// `ExecNoSync` is like `Exec`, except it embeds a reference to a cache. This
/// means it is no longer Sync, but we can now avoid the overhead of
/// synchronization to fetch the cache.
#[derive(Debug)]
pub struct ExecNoSync<'c> {
    /// All read only state.
    ro: &'c Arc<ExecReadOnly>,
    /// Caches for the various matching engines.
    cache: CachedGuard<'c, ProgramCache>,
}

/// `ExecNoSyncStr` is like `ExecNoSync`, but matches on &str instead of &[u8].
pub struct ExecNoSyncStr<'c>(ExecNoSync<'c>);

/// `ExecReadOnly` comprises all read only state for a regex. Namely, all such
/// state is determined at compile time and never changes during search.
#[derive(Debug)]
struct ExecReadOnly {
    /// The original regular expressions given by the caller to compile.
    res: Vec<String>,
    /// A compiled program that is used in the NFA simulation and backtracking.
    /// It can be byte-based or Unicode codepoint based.
    ///
    /// N.B. It is not possibly to make this byte-based from the public API.
    /// It is only used for testing byte based programs in the NFA simulations.
    nfa: Program,
    /// A compiled byte based program for DFA execution. This is only used
    /// if a DFA can be executed. (Currently, only word boundary assertions are
    /// not supported.) Note that this program contains an embedded `.*?`
    /// preceding the first capture group, unless the regex is anchored at the
    /// beginning.
    dfa: Program,
    /// The same as above, except the program is reversed (and there is no
    /// preceding `.*?`). This is used by the DFA to find the starting location
    /// of matches.
    dfa_reverse: Program,
    /// A set of suffix literals extracted from the regex.
    ///
    /// Prefix literals are stored on the `Program`, since they are used inside
    /// the matching engines.
    suffixes: LiteralSearcher,
    /// An Aho-Corasick automaton with leftmost-first match semantics.
    ///
    /// This is only set when the entire regex is a simple unanchored
    /// alternation of literals. We could probably use it more circumstances,
    /// but this is already hacky enough in this architecture.
    ///
    /// N.B. We use u32 as a state ID representation under the assumption that
    /// if we were to exhaust the ID space, we probably would have long
    /// surpassed the compilation size limit.
    #[cfg(feature = "perf-literal")]
    ac: Option<AhoCorasick<u32>>,
    /// match_type encodes as much upfront knowledge about how we're going to
    /// execute a search as possible.
    match_type: MatchType,
}

/// Facilitates the construction of an executor by exposing various knobs
/// to control how a regex is executed and what kinds of resources it's
/// permitted to use.
pub struct ExecBuilder {
    options: RegexOptions,
    match_type: Option<MatchType>,
    bytes: bool,
    only_utf8: bool,
}

/// Parsed represents a set of parsed regular expressions and their detected
/// literals.
struct Parsed {
    exprs: Vec<Hir>,
    prefixes: Literals,
    suffixes: Literals,
    bytes: bool,
}

impl ExecBuilder {
    /// Create a regex execution builder.
    ///
    /// This uses default settings for everything except the regex itself,
    /// which must be provided. Further knobs can be set by calling methods,
    /// and then finally, `build` to actually create the executor.
    pub fn new(re: &str) -> Self {
        Self::new_many(&[re])
    }

    /// Like new, but compiles the union of the given regular expressions.
    ///
    /// Note that when compiling 2 or more regular expressions, capture groups
    /// are completely unsupported. (This means both `find` and `captures`
    /// wont work.)
    pub fn new_many<I, S>(res: I) -> Self
    where
        S: AsRef<str>,
        I: IntoIterator<Item = S>,
    {
        let mut opts = RegexOptions::default();
        opts.pats = res.into_iter().map(|s| s.as_ref().to_owned()).collect();
        Self::new_options(opts)
    }

    /// Create a regex execution builder.
    pub fn new_options(opts: RegexOptions) -> Self {
        ExecBuilder {
            options: opts,
            match_type: None,
            bytes: false,
            only_utf8: true,
        }
    }

    /// Set the matching engine to be automatically determined.
    ///
    /// This is the default state and will apply whatever optimizations are
    /// possible, such as running a DFA.
    ///
    /// This overrides whatever was previously set via the `nfa` or
    /// `bounded_backtracking` methods.
    pub fn automatic(mut self) -> Self {
        self.match_type = None;
        self
    }

    /// Sets the matching engine to use the NFA algorithm no matter what
    /// optimizations are possible.
    ///
    /// This overrides whatever was previously set via the `automatic` or
    /// `bounded_backtracking` methods.
    pub fn nfa(mut self) -> Self {
        self.match_type = Some(MatchType::Nfa(MatchNfaType::PikeVM));
        self
    }

    /// Sets the matching engine to use a bounded backtracking engine no
    /// matter what optimizations are possible.
    ///
    /// One must use this with care, since the bounded backtracking engine
    /// uses memory proportion to `len(regex) * len(text)`.
    ///
    /// This overrides whatever was previously set via the `automatic` or
    /// `nfa` methods.
    pub fn bounded_backtracking(mut self) -> Self {
        self.match_type = Some(MatchType::Nfa(MatchNfaType::Backtrack));
        self
    }

    /// Compiles byte based programs for use with the NFA matching engines.
    ///
    /// By default, the NFA engines match on Unicode scalar values. They can
    /// be made to use byte based programs instead. In general, the byte based
    /// programs are slower because of a less efficient encoding of character
    /// classes.
    ///
    /// Note that this does not impact DFA matching engines, which always
    /// execute on bytes.
    pub fn bytes(mut self, yes: bool) -> Self {
        self.bytes = yes;
        self
    }

    /// When disabled, the program compiled may match arbitrary bytes.
    ///
    /// When enabled (the default), all compiled programs exclusively match
    /// valid UTF-8 bytes.
    pub fn only_utf8(mut self, yes: bool) -> Self {
        self.only_utf8 = yes;
        self
    }

    /// Set the Unicode flag.
    pub fn unicode(mut self, yes: bool) -> Self {
        self.options.unicode = yes;
        self
    }

    /// Parse the current set of patterns into their AST and extract literals.
    fn parse(&self) -> Result<Parsed, Error> {
        let mut exprs = Vec::with_capacity(self.options.pats.len());
        let mut prefixes = Some(Literals::empty());
        let mut suffixes = Some(Literals::empty());
        let mut bytes = false;
        let is_set = self.options.pats.len() > 1;
        // If we're compiling a regex set and that set has any anchored
        // expressions, then disable all literal optimizations.
        for pat in &self.options.pats {
            let mut parser = ParserBuilder::new()
                .octal(self.options.octal)
                .case_insensitive(self.options.case_insensitive)
                .multi_line(self.options.multi_line)
                .dot_matches_new_line(self.options.dot_matches_new_line)
                .swap_greed(self.options.swap_greed)
                .ignore_whitespace(self.options.ignore_whitespace)
                .unicode(self.options.unicode)
                .allow_invalid_utf8(!self.only_utf8)
                .nest_limit(self.options.nest_limit)
                .build();
            let expr =
                parser.parse(pat).map_err(|e| Error::Syntax(e.to_string()))?;
            bytes = bytes || !expr.is_always_utf8();

            if cfg!(feature = "perf-literal") {
                if !expr.is_anchored_start() && expr.is_any_anchored_start() {
                    // Partial anchors unfortunately make it hard to use
                    // prefixes, so disable them.
                    prefixes = None;
                } else if is_set && expr.is_anchored_start() {
                    // Regex sets with anchors do not go well with literal
                    // optimizations.
                    prefixes = None;
                }
                prefixes = prefixes.and_then(|mut prefixes| {
                    if !prefixes.union_prefixes(&expr) {
                        None
                    } else {
                        Some(prefixes)
                    }
                });

                if !expr.is_anchored_end() && expr.is_any_anchored_end() {
                    // Partial anchors unfortunately make it hard to use
                    // suffixes, so disable them.
                    suffixes = None;
                } else if is_set && expr.is_anchored_end() {
                    // Regex sets with anchors do not go well with literal
                    // optimizations.
                    suffixes = None;
                }
                suffixes = suffixes.and_then(|mut suffixes| {
                    if !suffixes.union_suffixes(&expr) {
                        None
                    } else {
                        Some(suffixes)
                    }
                });
            }
            exprs.push(expr);
        }
        Ok(Parsed {
            exprs: exprs,
            prefixes: prefixes.unwrap_or_else(Literals::empty),
            suffixes: suffixes.unwrap_or_else(Literals::empty),
            bytes: bytes,
        })
    }

    /// Build an executor that can run a regular expression.
    pub fn build(self) -> Result<Exec, Error> {
        // Special case when we have no patterns to compile.
        // This can happen when compiling a regex set.
        if self.options.pats.is_empty() {
            let ro = Arc::new(ExecReadOnly {
                res: vec![],
                nfa: Program::new(),
                dfa: Program::new(),
                dfa_reverse: Program::new(),
                suffixes: LiteralSearcher::empty(),
                #[cfg(feature = "perf-literal")]
                ac: None,
                match_type: MatchType::Nothing,
            });
            return Ok(Exec { ro: ro, cache: Cached::new() });
        }
        let parsed = self.parse()?;
        let mut nfa = Compiler::new()
            .size_limit(self.options.size_limit)
            .bytes(self.bytes || parsed.bytes)
            .only_utf8(self.only_utf8)
            .compile(&parsed.exprs)?;
        let mut dfa = Compiler::new()
            .size_limit(self.options.size_limit)
            .dfa(true)
            .only_utf8(self.only_utf8)
            .compile(&parsed.exprs)?;
        let mut dfa_reverse = Compiler::new()
            .size_limit(self.options.size_limit)
            .dfa(true)
            .only_utf8(self.only_utf8)
            .reverse(true)
            .compile(&parsed.exprs)?;

        #[cfg(feature = "perf-literal")]
        let ac = self.build_aho_corasick(&parsed);
        nfa.prefixes = LiteralSearcher::prefixes(parsed.prefixes);
        dfa.prefixes = nfa.prefixes.clone();
        dfa.dfa_size_limit = self.options.dfa_size_limit;
        dfa_reverse.dfa_size_limit = self.options.dfa_size_limit;

        let mut ro = ExecReadOnly {
            res: self.options.pats,
            nfa: nfa,
            dfa: dfa,
            dfa_reverse: dfa_reverse,
            suffixes: LiteralSearcher::suffixes(parsed.suffixes),
            #[cfg(feature = "perf-literal")]
            ac: ac,
            match_type: MatchType::Nothing,
        };
        ro.match_type = ro.choose_match_type(self.match_type);

        let ro = Arc::new(ro);
        Ok(Exec { ro: ro, cache: Cached::new() })
    }

    #[cfg(feature = "perf-literal")]
    fn build_aho_corasick(&self, parsed: &Parsed) -> Option<AhoCorasick<u32>> {
        if parsed.exprs.len() != 1 {
            return None;
        }
        let lits = match alternation_literals(&parsed.exprs[0]) {
            None => return None,
            Some(lits) => lits,
        };
        // If we have a small number of literals, then let Teddy handle
        // things (see literal/mod.rs).
        if lits.len() <= 32 {
            return None;
        }
        Some(
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostFirst)
                .auto_configure(&lits)
                // We always want this to reduce size, regardless
                // of what auto-configure does.
                .byte_classes(true)
                .build_with_size::<u32, _, _>(&lits)
                // This should never happen because we'd long exceed the
                // compilation limit for regexes first.
                .expect("AC automaton too big"),
        )
    }
}

impl<'c> RegularExpression for ExecNoSyncStr<'c> {
    type Text = str;

    fn slots_len(&self) -> usize {
        self.0.slots_len()
    }

    fn next_after_empty(&self, text: &str, i: usize) -> usize {
        next_utf8(text.as_bytes(), i)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_match_at(&self, text: &str, start: usize) -> Option<usize> {
        self.0.shortest_match_at(text.as_bytes(), start)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_match_at(&self, text: &str, start: usize) -> bool {
        self.0.is_match_at(text.as_bytes(), start)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_at(&self, text: &str, start: usize) -> Option<(usize, usize)> {
        self.0.find_at(text.as_bytes(), start)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn captures_read_at(
        &self,
        locs: &mut Locations,
        text: &str,
        start: usize,
    ) -> Option<(usize, usize)> {
        self.0.captures_read_at(locs, text.as_bytes(), start)
    }
}

impl<'c> RegularExpression for ExecNoSync<'c> {
    type Text = [u8];

    /// Returns the number of capture slots in the regular expression. (There
    /// are two slots for every capture group, corresponding to possibly empty
    /// start and end locations of the capture.)
    fn slots_len(&self) -> usize {
        self.ro.nfa.captures.len() * 2
    }

    fn next_after_empty(&self, _text: &[u8], i: usize) -> usize {
        i + 1
    }

    /// Returns the end of a match location, possibly occurring before the
    /// end location of the correct leftmost-first match.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_match_at(&self, text: &[u8], start: usize) -> Option<usize> {
        if !self.is_anchor_end_match(text) {
            return None;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => {
                self.find_literals(ty, text, start).map(|(_, e)| e)
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa | MatchType::DfaMany => {
                match self.shortest_dfa(text, start) {
                    dfa::Result::Match(end) => Some(end),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.shortest_nfa(text, start),
                }
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match dfa::Fsm::reverse(
                    &self.ro.dfa_reverse,
                    self.cache.value(),
                    true,
                    &text[start..],
                    text.len(),
                ) {
                    dfa::Result::Match(_) => Some(text.len()),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.shortest_nfa(text, start),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.shortest_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match(e) => Some(e),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.shortest_nfa(text, start),
                }
            }
            MatchType::Nfa(ty) => self.shortest_nfa_type(ty, text, start),
            MatchType::Nothing => None,
        }
    }

    /// Returns true if and only if the regex matches text.
    ///
    /// For single regular expressions, this is equivalent to calling
    /// shortest_match(...).is_some().
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_match_at(&self, text: &[u8], start: usize) -> bool {
        if !self.is_anchor_end_match(text) {
            return false;
        }
        // We need to do this dance because shortest_match relies on the NFA
        // filling in captures[1], but a RegexSet has no captures. In other
        // words, a RegexSet can't (currently) use shortest_match. ---AG
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => {
                self.find_literals(ty, text, start).is_some()
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa | MatchType::DfaMany => {
                match self.shortest_dfa(text, start) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.match_nfa(text, start),
                }
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match dfa::Fsm::reverse(
                    &self.ro.dfa_reverse,
                    self.cache.value(),
                    true,
                    &text[start..],
                    text.len(),
                ) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.match_nfa(text, start),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.shortest_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.match_nfa(text, start),
                }
            }
            MatchType::Nfa(ty) => self.match_nfa_type(ty, text, start),
            MatchType::Nothing => false,
        }
    }

    /// Finds the start and end location of the leftmost-first match, starting
    /// at the given location.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_at(&self, text: &[u8], start: usize) -> Option<(usize, usize)> {
        if !self.is_anchor_end_match(text) {
            return None;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => self.find_literals(ty, text, start),
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa => match self.find_dfa_forward(text, start) {
                dfa::Result::Match((s, e)) => Some((s, e)),
                dfa::Result::NoMatch(_) => None,
                dfa::Result::Quit => {
                    self.find_nfa(MatchNfaType::Auto, text, start)
                }
            },
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match self.find_dfa_anchored_reverse(text, start) {
                    dfa::Result::Match((s, e)) => Some((s, e)),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => {
                        self.find_nfa(MatchNfaType::Auto, text, start)
                    }
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.find_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match((s, e)) => Some((s, e)),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => {
                        self.find_nfa(MatchNfaType::Auto, text, start)
                    }
                }
            }
            MatchType::Nfa(ty) => self.find_nfa(ty, text, start),
            MatchType::Nothing => None,
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaMany => {
                unreachable!("BUG: RegexSet cannot be used with find")
            }
        }
    }

    /// Finds the start and end location of the leftmost-first match and also
    /// fills in all matching capture groups.
    ///
    /// The number of capture slots given should be equal to the total number
    /// of capture slots in the compiled program.
    ///
    /// Note that the first two slots always correspond to the start and end
    /// locations of the overall match.
    fn captures_read_at(
        &self,
        locs: &mut Locations,
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        let slots = locs.as_slots();
        for slot in slots.iter_mut() {
            *slot = None;
        }
        // If the caller unnecessarily uses this, then we try to save them
        // from themselves.
        match slots.len() {
            0 => return self.find_at(text, start),
            2 => {
                return self.find_at(text, start).map(|(s, e)| {
                    slots[0] = Some(s);
                    slots[1] = Some(e);
                    (s, e)
                });
            }
            _ => {} // fallthrough
        }
        if !self.is_anchor_end_match(text) {
            return None;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => {
                self.find_literals(ty, text, start).and_then(|(s, e)| {
                    self.captures_nfa_type(
                        MatchNfaType::Auto,
                        slots,
                        text,
                        s,
                        e,
                    )
                })
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa => {
                if self.ro.nfa.is_anchored_start {
                    self.captures_nfa(slots, text, start)
                } else {
                    match self.find_dfa_forward(text, start) {
                        dfa::Result::Match((s, e)) => self.captures_nfa_type(
                            MatchNfaType::Auto,
                            slots,
                            text,
                            s,
                            e,
                        ),
                        dfa::Result::NoMatch(_) => None,
                        dfa::Result::Quit => {
                            self.captures_nfa(slots, text, start)
                        }
                    }
                }
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match self.find_dfa_anchored_reverse(text, start) {
                    dfa::Result::Match((s, e)) => self.captures_nfa_type(
                        MatchNfaType::Auto,
                        slots,
                        text,
                        s,
                        e,
                    ),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.captures_nfa(slots, text, start),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.find_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match((s, e)) => self.captures_nfa_type(
                        MatchNfaType::Auto,
                        slots,
                        text,
                        s,
                        e,
                    ),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.captures_nfa(slots, text, start),
                }
            }
            MatchType::Nfa(ty) => {
                self.captures_nfa_type(ty, slots, text, start, text.len())
            }
            MatchType::Nothing => None,
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaMany => {
                unreachable!("BUG: RegexSet cannot be used with captures")
            }
        }
    }
}

impl<'c> ExecNoSync<'c> {
    /// Finds the leftmost-first match using only literal search.
    #[cfg(feature = "perf-literal")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_literals(
        &self,
        ty: MatchLiteralType,
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        use self::MatchLiteralType::*;
        match ty {
            Unanchored => {
                let lits = &self.ro.nfa.prefixes;
                lits.find(&text[start..]).map(|(s, e)| (start + s, start + e))
            }
            AnchoredStart => {
                let lits = &self.ro.nfa.prefixes;
                if start == 0 || !self.ro.nfa.is_anchored_start {
                    lits.find_start(&text[start..])
                        .map(|(s, e)| (start + s, start + e))
                } else {
                    None
                }
            }
            AnchoredEnd => {
                let lits = &self.ro.suffixes;
                lits.find_end(&text[start..])
                    .map(|(s, e)| (start + s, start + e))
            }
            AhoCorasick => self
                .ro
                .ac
                .as_ref()
                .unwrap()
                .find(&text[start..])
                .map(|m| (start + m.start(), start + m.end())),
        }
    }

    /// Finds the leftmost-first match (start and end) using only the DFA.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(feature = "perf-dfa")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_dfa_forward(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<(usize, usize)> {
        use dfa::Result::*;
        let end = match dfa::Fsm::forward(
            &self.ro.dfa,
            self.cache.value(),
            false,
            text,
            start,
        ) {
            NoMatch(i) => return NoMatch(i),
            Quit => return Quit,
            Match(end) if start == end => return Match((start, start)),
            Match(end) => end,
        };
        // Now run the DFA in reverse to find the start of the match.
        match dfa::Fsm::reverse(
            &self.ro.dfa_reverse,
            self.cache.value(),
            false,
            &text[start..],
            end - start,
        ) {
            Match(s) => Match((start + s, end)),
            NoMatch(i) => NoMatch(i),
            Quit => Quit,
        }
    }

    /// Finds the leftmost-first match (start and end) using only the DFA,
    /// but assumes the regex is anchored at the end and therefore starts at
    /// the end of the regex and matches in reverse.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(feature = "perf-dfa")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_dfa_anchored_reverse(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<(usize, usize)> {
        use dfa::Result::*;
        match dfa::Fsm::reverse(
            &self.ro.dfa_reverse,
            self.cache.value(),
            false,
            &text[start..],
            text.len() - start,
        ) {
            Match(s) => Match((start + s, text.len())),
            NoMatch(i) => NoMatch(i),
            Quit => Quit,
        }
    }

    /// Finds the end of the shortest match using only the DFA.
    #[cfg(feature = "perf-dfa")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_dfa(&self, text: &[u8], start: usize) -> dfa::Result<usize> {
        dfa::Fsm::forward(&self.ro.dfa, self.cache.value(), true, text, start)
    }

    /// Finds the end of the shortest match using only the DFA by scanning for
    /// suffix literals.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_dfa_reverse_suffix(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<usize> {
        match self.exec_dfa_reverse_suffix(text, start) {
            None => self.shortest_dfa(text, start),
            Some(r) => r.map(|(_, end)| end),
        }
    }

    /// Finds the end of the shortest match using only the DFA by scanning for
    /// suffix literals. It also reports the start of the match.
    ///
    /// Note that if None is returned, then the optimization gave up to avoid
    /// worst case quadratic behavior. A forward scanning DFA should be tried
    /// next.
    ///
    /// If a match is returned and the full leftmost-first match is desired,
    /// then a forward scan starting from the beginning of the match must be
    /// done.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn exec_dfa_reverse_suffix(
        &self,
        text: &[u8],
        original_start: usize,
    ) -> Option<dfa::Result<(usize, usize)>> {
        use dfa::Result::*;

        let lcs = self.ro.suffixes.lcs();
        debug_assert!(lcs.len() >= 1);
        let mut start = original_start;
        let mut end = start;
        let mut last_literal = start;
        while end <= text.len() {
            last_literal += match lcs.find(&text[last_literal..]) {
                None => return Some(NoMatch(text.len())),
                Some(i) => i,
            };
            end = last_literal + lcs.len();
            match dfa::Fsm::reverse(
                &self.ro.dfa_reverse,
                self.cache.value(),
                false,
                &text[start..end],
                end - start,
            ) {
                Match(0) | NoMatch(0) => return None,
                Match(i) => return Some(Match((start + i, end))),
                NoMatch(i) => {
                    start += i;
                    last_literal += 1;
                    continue;
                }
                Quit => return Some(Quit),
            };
        }
        Some(NoMatch(text.len()))
    }

    /// Finds the leftmost-first match (start and end) using only the DFA
    /// by scanning for suffix literals.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_dfa_reverse_suffix(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<(usize, usize)> {
        use dfa::Result::*;

        let match_start = match self.exec_dfa_reverse_suffix(text, start) {
            None => return self.find_dfa_forward(text, start),
            Some(Match((start, _))) => start,
            Some(r) => return r,
        };
        // At this point, we've found a match. The only way to quit now
        // without a match is if the DFA gives up (seems unlikely).
        //
        // Now run the DFA forwards to find the proper end of the match.
        // (The suffix literal match can only indicate the earliest
        // possible end location, which may appear before the end of the
        // leftmost-first match.)
        match dfa::Fsm::forward(
            &self.ro.dfa,
            self.cache.value(),
            false,
            text,
            match_start,
        ) {
            NoMatch(_) => panic!("BUG: reverse match implies forward match"),
            Quit => Quit,
            Match(e) => Match((match_start, e)),
        }
    }

    /// Executes the NFA engine to return whether there is a match or not.
    ///
    /// Ideally, we could use shortest_nfa(...).is_some() and get the same
    /// performance characteristics, but regex sets don't have captures, which
    /// shortest_nfa depends on.
    #[cfg(feature = "perf-dfa")]
    fn match_nfa(&self, text: &[u8], start: usize) -> bool {
        self.match_nfa_type(MatchNfaType::Auto, text, start)
    }

    /// Like match_nfa, but allows specification of the type of NFA engine.
    fn match_nfa_type(
        &self,
        ty: MatchNfaType,
        text: &[u8],
        start: usize,
    ) -> bool {
        self.exec_nfa(
            ty,
            &mut [false],
            &mut [],
            true,
            false,
            text,
            start,
            text.len(),
        )
    }

    /// Finds the shortest match using an NFA.
    #[cfg(feature = "perf-dfa")]
    fn shortest_nfa(&self, text: &[u8], start: usize) -> Option<usize> {
        self.shortest_nfa_type(MatchNfaType::Auto, text, start)
    }

    /// Like shortest_nfa, but allows specification of the type of NFA engine.
    fn shortest_nfa_type(
        &self,
        ty: MatchNfaType,
        text: &[u8],
        start: usize,
    ) -> Option<usize> {
        let mut slots = [None, None];
        if self.exec_nfa(
            ty,
            &mut [false],
            &mut slots,
            true,
            true,
            text,
            start,
            text.len(),
        ) {
            slots[1]
        } else {
            None
        }
    }

    /// Like find, but executes an NFA engine.
    fn find_nfa(
        &self,
        ty: MatchNfaType,
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        let mut slots = [None, None];
        if self.exec_nfa(
            ty,
            &mut [false],
            &mut slots,
            false,
            false,
            text,
            start,
            text.len(),
        ) {
            match (slots[0], slots[1]) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Like find_nfa, but fills in captures.
    ///
    /// `slots` should have length equal to `2 * nfa.captures.len()`.
    #[cfg(feature = "perf-dfa")]
    fn captures_nfa(
        &self,
        slots: &mut [Slot],
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        self.captures_nfa_type(
            MatchNfaType::Auto,
            slots,
            text,
            start,
            text.len(),
        )
    }

    /// Like captures_nfa, but allows specification of type of NFA engine.
    fn captures_nfa_type(
        &self,
        ty: MatchNfaType,
        slots: &mut [Slot],
        text: &[u8],
        start: usize,
        end: usize,
    ) -> Option<(usize, usize)> {
        if self.exec_nfa(
            ty,
            &mut [false],
            slots,
            false,
            false,
            text,
            start,
            end,
        ) {
            match (slots[0], slots[1]) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            }
        } else {
            None
        }
    }

    fn exec_nfa(
        &self,
        mut ty: MatchNfaType,
        matches: &mut [bool],
        slots: &mut [Slot],
        quit_after_match: bool,
        quit_after_match_with_pos: bool,
        text: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        use self::MatchNfaType::*;
        if let Auto = ty {
            if backtrack::should_exec(self.ro.nfa.len(), text.len()) {
                ty = Backtrack;
            } else {
                ty = PikeVM;
            }
        }
        // The backtracker can't return the shortest match position as it is
        // implemented today. So if someone calls `shortest_match` and we need
        // to run an NFA, then use the PikeVM.
        if quit_after_match_with_pos || ty == PikeVM {
            self.exec_pikevm(
                matches,
                slots,
                quit_after_match,
                text,
                start,
                end,
            )
        } else {
            self.exec_backtrack(matches, slots, text, start, end)
        }
    }

    /// Always run the NFA algorithm.
    fn exec_pikevm(
        &self,
        matches: &mut [bool],
        slots: &mut [Slot],
        quit_after_match: bool,
        text: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        if self.ro.nfa.uses_bytes() {
            pikevm::Fsm::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                quit_after_match,
                ByteInput::new(text, self.ro.nfa.only_utf8),
                start,
                end,
            )
        } else {
            pikevm::Fsm::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                quit_after_match,
                CharInput::new(text),
                start,
                end,
            )
        }
    }

    /// Always runs the NFA using bounded backtracking.
    fn exec_backtrack(
        &self,
        matches: &mut [bool],
        slots: &mut [Slot],
        text: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        if self.ro.nfa.uses_bytes() {
            backtrack::Bounded::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                ByteInput::new(text, self.ro.nfa.only_utf8),
                start,
                end,
            )
        } else {
            backtrack::Bounded::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                CharInput::new(text),
                start,
                end,
            )
        }
    }

    /// Finds which regular expressions match the given text.
    ///
    /// `matches` should have length equal to the number of regexes being
    /// searched.
    ///
    /// This is only useful when one wants to know which regexes in a set
    /// match some text.
    pub fn many_matches_at(
        &self,
        matches: &mut [bool],
        text: &[u8],
        start: usize,
    ) -> bool {
        use self::MatchType::*;
        if !self.is_anchor_end_match(text) {
            return false;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            Literal(ty) => {
                debug_assert_eq!(matches.len(), 1);
                matches[0] = self.find_literals(ty, text, start).is_some();
                matches[0]
            }
            #[cfg(feature = "perf-dfa")]
            Dfa | DfaAnchoredReverse | DfaMany => {
                match dfa::Fsm::forward_many(
                    &self.ro.dfa,
                    self.cache.value(),
                    matches,
                    text,
                    start,
                ) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.exec_nfa(
                        MatchNfaType::Auto,
                        matches,
                        &mut [],
                        false,
                        false,
                        text,
                        start,
                        text.len(),
                    ),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            DfaSuffix => {
                match dfa::Fsm::forward_many(
                    &self.ro.dfa,
                    self.cache.value(),
                    matches,
                    text,
                    start,
                ) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.exec_nfa(
                        MatchNfaType::Auto,
                        matches,
                        &mut [],
                        false,
                        false,
                        text,
                        start,
                        text.len(),
                    ),
                }
            }
            Nfa(ty) => self.exec_nfa(
                ty,
                matches,
                &mut [],
                false,
                false,
                text,
                start,
                text.len(),
            ),
            Nothing => false,
        }
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_anchor_end_match(&self, text: &[u8]) -> bool {
        #[cfg(not(feature = "perf-literal"))]
        fn imp(_: &ExecReadOnly, _: &[u8]) -> bool {
            true
        }

        #[cfg(feature = "perf-literal")]
        fn imp(ro: &ExecReadOnly, text: &[u8]) -> bool {
            // Only do this check if the haystack is big (>1MB).
            if text.len() > (1 << 20) && ro.nfa.is_anchored_end {
                let lcs = ro.suffixes.lcs();
                if lcs.len() >= 1 && !lcs.is_suffix(text) {
                    return false;
                }
            }
            true
        }

        imp(&self.ro, text)
    }

    pub fn capture_name_idx(&self) -> &Arc<HashMap<String, usize>> {
        &self.ro.nfa.capture_name_idx
    }
}

impl<'c> ExecNoSyncStr<'c> {
    pub fn capture_name_idx(&self) -> &Arc<HashMap<String, usize>> {
        self.0.capture_name_idx()
    }
}

impl Exec {
    /// Get a searcher that isn't Sync.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn searcher(&self) -> ExecNoSync {
        let create = || RefCell::new(ProgramCacheInner::new(&self.ro));
        ExecNoSync {
            ro: &self.ro, // a clone is too expensive here! (and not needed)
            cache: self.cache.get_or(create),
        }
    }

    /// Get a searcher that isn't Sync and can match on &str.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn searcher_str(&self) -> ExecNoSyncStr {
        ExecNoSyncStr(self.searcher())
    }

    /// Build a Regex from this executor.
    pub fn into_regex(self) -> re_unicode::Regex {
        re_unicode::Regex::from(self)
    }

    /// Build a RegexSet from this executor.
    pub fn into_regex_set(self) -> re_set::unicode::RegexSet {
        re_set::unicode::RegexSet::from(self)
    }

    /// Build a Regex from this executor that can match arbitrary bytes.
    pub fn into_byte_regex(self) -> re_bytes::Regex {
        re_bytes::Regex::from(self)
    }

    /// Build a RegexSet from this executor that can match arbitrary bytes.
    pub fn into_byte_regex_set(self) -> re_set::bytes::RegexSet {
        re_set::bytes::RegexSet::from(self)
    }

    /// The original regular expressions given by the caller that were
    /// compiled.
    pub fn regex_strings(&self) -> &[String] {
        &self.ro.res
    }

    /// Return a slice of capture names.
    ///
    /// Any capture that isn't named is None.
    pub fn capture_names(&self) -> &[Option<String>] {
        &self.ro.nfa.captures
    }

    /// Return a reference to named groups mapping (from group name to
    /// group position).
    pub fn capture_name_idx(&self) -> &Arc<HashMap<String, usize>> {
        &self.ro.nfa.capture_name_idx
    }
}

impl Clone for Exec {
    fn clone(&self) -> Exec {
        Exec { ro: self.ro.clone(), cache: Cached::new() }
    }
}

impl ExecReadOnly {
    fn choose_match_type(&self, hint: Option<MatchType>) -> MatchType {
        if let Some(MatchType::Nfa(_)) = hint {
            return hint.unwrap();
        }
        // If the NFA is empty, then we'll never match anything.
        if self.nfa.insts.is_empty() {
            return MatchType::Nothing;
        }
        if let Some(literalty) = self.choose_literal_match_type() {
            return literalty;
        }
        if let Some(dfaty) = self.choose_dfa_match_type() {
            return dfaty;
        }
        // We're so totally hosed.
        MatchType::Nfa(MatchNfaType::Auto)
    }

    /// If a plain literal scan can be used, then a corresponding literal
    /// search type is returned.
    fn choose_literal_match_type(&self) -> Option<MatchType> {
        #[cfg(not(feature = "perf-literal"))]
        fn imp(_: &ExecReadOnly) -> Option<MatchType> {
            None
        }

        #[cfg(feature = "perf-literal")]
        fn imp(ro: &ExecReadOnly) -> Option<MatchType> {
            // If our set of prefixes is complete, then we can use it to find
            // a match in lieu of a regex engine. This doesn't quite work well
            // in the presence of multiple regexes, so only do it when there's
            // one.
            //
            // TODO(burntsushi): Also, don't try to match literals if the regex
            // is partially anchored. We could technically do it, but we'd need
            // to create two sets of literals: all of them and then the subset
            // that aren't anchored. We would then only search for all of them
            // when at the beginning of the input and use the subset in all
            // other cases.
            if ro.res.len() != 1 {
                return None;
            }
            if ro.ac.is_some() {
                return Some(MatchType::Literal(
                    MatchLiteralType::AhoCorasick,
                ));
            }
            if ro.nfa.prefixes.complete() {
                return if ro.nfa.is_anchored_start {
                    Some(MatchType::Literal(MatchLiteralType::AnchoredStart))
                } else {
                    Some(MatchType::Literal(MatchLiteralType::Unanchored))
                };
            }
            if ro.suffixes.complete() {
                return if ro.nfa.is_anchored_end {
                    Some(MatchType::Literal(MatchLiteralType::AnchoredEnd))
                } else {
                    // This case shouldn't happen. When the regex isn't
                    // anchored, then complete prefixes should imply complete
                    // suffixes.
                    Some(MatchType::Literal(MatchLiteralType::Unanchored))
                };
            }
            None
        }

        imp(self)
    }

    /// If a DFA scan can be used, then choose the appropriate DFA strategy.
    fn choose_dfa_match_type(&self) -> Option<MatchType> {
        #[cfg(not(feature = "perf-dfa"))]
        fn imp(_: &ExecReadOnly) -> Option<MatchType> {
            None
        }

        #[cfg(feature = "perf-dfa")]
        fn imp(ro: &ExecReadOnly) -> Option<MatchType> {
            if !dfa::can_exec(&ro.dfa) {
                return None;
            }
            // Regex sets require a slightly specialized path.
            if ro.res.len() >= 2 {
                return Some(MatchType::DfaMany);
            }
            // If the regex is anchored at the end but not the start, then
            // just match in reverse from the end of the haystack.
            if !ro.nfa.is_anchored_start && ro.nfa.is_anchored_end {
                return Some(MatchType::DfaAnchoredReverse);
            }
            #[cfg(feature = "perf-literal")]
            {
                // If there's a longish suffix literal, then it might be faster
                // to look for that first.
                if ro.should_suffix_scan() {
                    return Some(MatchType::DfaSuffix);
                }
            }
            // Fall back to your garden variety forward searching lazy DFA.
            Some(MatchType::Dfa)
        }

        imp(self)
    }

    /// Returns true if the program is amenable to suffix scanning.
    ///
    /// When this is true, as a heuristic, we assume it is OK to quickly scan
    /// for suffix literals and then do a *reverse* DFA match from any matches
    /// produced by the literal scan. (And then followed by a forward DFA
    /// search, since the previously found suffix literal maybe not actually be
    /// the end of a match.)
    ///
    /// This is a bit of a specialized optimization, but can result in pretty
    /// big performance wins if 1) there are no prefix literals and 2) the
    /// suffix literals are pretty rare in the text. (1) is obviously easy to
    /// account for but (2) is harder. As a proxy, we assume that longer
    /// strings are generally rarer, so we only enable this optimization when
    /// we have a meaty suffix.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    fn should_suffix_scan(&self) -> bool {
        if self.suffixes.is_empty() {
            return false;
        }
        let lcs_len = self.suffixes.lcs().char_len();
        lcs_len >= 3 && lcs_len > self.dfa.prefixes.lcp().char_len()
    }
}

#[derive(Clone, Copy, Debug)]
enum MatchType {
    /// A single or multiple literal search. This is only used when the regex
    /// can be decomposed into a literal search.
    #[cfg(feature = "perf-literal")]
    Literal(MatchLiteralType),
    /// A normal DFA search.
    #[cfg(feature = "perf-dfa")]
    Dfa,
    /// A reverse DFA search starting from the end of a haystack.
    #[cfg(feature = "perf-dfa")]
    DfaAnchoredReverse,
    /// A reverse DFA search with suffix literal scanning.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    DfaSuffix,
    /// Use the DFA on two or more regular expressions.
    #[cfg(feature = "perf-dfa")]
    DfaMany,
    /// An NFA variant.
    Nfa(MatchNfaType),
    /// No match is ever possible, so don't ever try to search.
    Nothing,
}

#[derive(Clone, Copy, Debug)]
#[cfg(feature = "perf-literal")]
enum MatchLiteralType {
    /// Match literals anywhere in text.
    Unanchored,
    /// Match literals only at the start of text.
    AnchoredStart,
    /// Match literals only at the end of text.
    AnchoredEnd,
    /// Use an Aho-Corasick automaton. This requires `ac` to be Some on
    /// ExecReadOnly.
    AhoCorasick,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatchNfaType {
    /// Choose between Backtrack and PikeVM.
    Auto,
    /// NFA bounded backtracking.
    ///
    /// (This is only set by tests, since it never makes sense to always want
    /// backtracking.)
    Backtrack,
    /// The Pike VM.
    ///
    /// (This is only set by tests, since it never makes sense to always want
    /// the Pike VM.)
    PikeVM,
}

/// `ProgramCache` maintains reusable allocations for each matching engine
/// available to a particular program.
pub type ProgramCache = RefCell<ProgramCacheInner>;

#[derive(Debug)]
pub struct ProgramCacheInner {
    pub pikevm: pikevm::Cache,
    pub backtrack: backtrack::Cache,
    #[cfg(feature = "perf-dfa")]
    pub dfa: dfa::Cache,
    #[cfg(feature = "perf-dfa")]
    pub dfa_reverse: dfa::Cache,
}

impl ProgramCacheInner {
    fn new(ro: &ExecReadOnly) -> Self {
        ProgramCacheInner {
            pikevm: pikevm::Cache::new(&ro.nfa),
            backtrack: backtrack::Cache::new(&ro.nfa),
            #[cfg(feature = "perf-dfa")]
            dfa: dfa::Cache::new(&ro.dfa),
            #[cfg(feature = "perf-dfa")]
            dfa_reverse: dfa::Cache::new(&ro.dfa_reverse),
        }
    }
}

/// Alternation literals checks if the given HIR is a simple alternation of
/// literals, and if so, returns them. Otherwise, this returns None.
#[cfg(feature = "perf-literal")]
fn alternation_literals(expr: &Hir) -> Option<Vec<Vec<u8>>> {
    use syntax::hir::{HirKind, Literal};

    // This is pretty hacky, but basically, if `is_alternation_literal` is
    // true, then we can make several assumptions about the structure of our
    // HIR. This is what justifies the `unreachable!` statements below.
    //
    // This code should be refactored once we overhaul this crate's
    // optimization pipeline, because this is a terribly inflexible way to go
    // about things.

    if !expr.is_alternation_literal() {
        return None;
    }
    let alts = match *expr.kind() {
        HirKind::Alternation(ref alts) => alts,
        _ => return None, // one literal isn't worth it
    };

    let extendlit = |lit: &Literal, dst: &mut Vec<u8>| match *lit {
        Literal::Unicode(c) => {
            let mut buf = [0; 4];
            dst.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
        Literal::Byte(b) => {
            dst.push(b);
        }
    };

    let mut lits = vec![];
    for alt in alts {
        let mut lit = vec![];
        match *alt.kind() {
            HirKind::Literal(ref x) => extendlit(x, &mut lit),
            HirKind::Concat(ref exprs) => {
                for e in exprs {
                    match *e.kind() {
                        HirKind::Literal(ref x) => extendlit(x, &mut lit),
                        _ => unreachable!("expected literal, got {:?}", e),
                    }
                }
            }
            _ => unreachable!("expected literal or concat, got {:?}", alt),
        }
        lits.push(lit);
    }
    Some(lits)
}

#[cfg(test)]
mod test {
    #[test]
    fn uppercut_s_backtracking_bytes_default_bytes_mismatch() {
        use internal::ExecBuilder;

        let backtrack_bytes_re = ExecBuilder::new("^S")
            .bounded_backtracking()
            .only_utf8(false)
            .build()
            .map(|exec| exec.into_byte_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let default_bytes_re = ExecBuilder::new("^S")
            .only_utf8(false)
            .build()
            .map(|exec| exec.into_byte_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let input = vec![83, 83];

        let s1 = backtrack_bytes_re.split(&input);
        let s2 = default_bytes_re.split(&input);
        for (chunk1, chunk2) in s1.zip(s2) {
            assert_eq!(chunk1, chunk2);
        }
    }

    #[test]
    fn unicode_lit_star_backtracking_utf8bytes_default_utf8bytes_mismatch() {
        use internal::ExecBuilder;

        let backtrack_bytes_re = ExecBuilder::new(r"^(?u:\*)")
            .bounded_backtracking()
            .bytes(true)
            .build()
            .map(|exec| exec.into_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let default_bytes_re = ExecBuilder::new(r"^(?u:\*)")
            .bytes(true)
            .build()
            .map(|exec| exec.into_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let input = "**";

        let s1 = backtrack_bytes_re.split(input);
        let s2 = default_bytes_re.split(input);
        for (chunk1, chunk2) in s1.zip(s2) {
            assert_eq!(chunk1, chunk2);
        }
    }
}
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::slice;
use std::sync::Arc;

use input::Char;
use literal::LiteralSearcher;

/// `InstPtr` represents the index of an instruction in a regex program.
pub type InstPtr = usize;

/// Program is a sequence of instructions and various facts about thos
/// instructions.
#[derive(Clone)]
pub struct Program {
    /// A sequence of instructions that represents an NFA.
    pub insts: Vec<Inst>,
    /// Pointers to each Match instruction in the sequence.
    ///
    /// This is always length 1 unless this program represents a regex set.
    pub matches: Vec<InstPtr>,
    /// The ordered sequence of all capture groups extracted from the AST.
    /// Unnamed groups are `None`.
    pub captures: Vec<Option<String>>,
    /// Pointers to all named capture groups into `captures`.
    pub capture_name_idx: Arc<HashMap<String, usize>>,
    /// A pointer to the start instruction. This can vary depending on how
    /// the program was compiled. For example, programs for use with the DFA
    /// engine have a `.*?` inserted at the beginning of unanchored regular
    /// expressions. The actual starting point of the program is after the
    /// `.*?`.
    pub start: InstPtr,
    /// A set of equivalence classes for discriminating bytes in the compiled
    /// program.
    pub byte_classes: Vec<u8>,
    /// When true, this program can only match valid UTF-8.
    pub only_utf8: bool,
    /// When true, this program uses byte range instructions instead of Unicode
    /// range instructions.
    pub is_bytes: bool,
    /// When true, the program is compiled for DFA matching. For example, this
    /// implies `is_bytes` and also inserts a preceding `.*?` for unanchored
    /// regexes.
    pub is_dfa: bool,
    /// When true, the program matches text in reverse (for use only in the
    /// DFA).
    pub is_reverse: bool,
    /// Whether the regex must match from the start of the input.
    pub is_anchored_start: bool,
    /// Whether the regex must match at the end of the input.
    pub is_anchored_end: bool,
    /// Whether this program contains a Unicode word boundary instruction.
    pub has_unicode_word_boundary: bool,
    /// A possibly empty machine for very quickly matching prefix literals.
    pub prefixes: LiteralSearcher,
    /// A limit on the size of the cache that the DFA is allowed to use while
    /// matching.
    ///
    /// The cache limit specifies approximately how much space we're willing to
    /// give to the state cache. Once the state cache exceeds the size, it is
    /// wiped and all states must be re-computed.
    ///
    /// Note that this value does not impact correctness. It can be set to 0
    /// and the DFA will run just fine. (It will only ever store exactly one
    /// state in the cache, and will likely run very slowly, but it will work.)
    ///
    /// Also note that this limit is *per thread of execution*. That is,
    /// if the same regex is used to search text across multiple threads
    /// simultaneously, then the DFA cache is not shared. Instead, copies are
    /// made.
    pub dfa_size_limit: usize,
}

impl Program {
    /// Creates an empty instruction sequence. Fields are given default
    /// values.
    pub fn new() -> Self {
        Program {
            insts: vec![],
            matches: vec![],
            captures: vec![],
            capture_name_idx: Arc::new(HashMap::new()),
            start: 0,
            byte_classes: vec![0; 256],
            only_utf8: true,
            is_bytes: false,
            is_dfa: false,
            is_reverse: false,
            is_anchored_start: false,
            is_anchored_end: false,
            has_unicode_word_boundary: false,
            prefixes: LiteralSearcher::empty(),
            dfa_size_limit: 2 * (1 << 20),
        }
    }

    /// If pc is an index to a no-op instruction (like Save), then return the
    /// next pc that is not a no-op instruction.
    pub fn skip(&self, mut pc: usize) -> usize {
        loop {
            match self[pc] {
                Inst::Save(ref i) => pc = i.goto,
                _ => return pc,
            }
        }
    }

    /// Return true if and only if an execution engine at instruction `pc` will
    /// always lead to a match.
    pub fn leads_to_match(&self, pc: usize) -> bool {
        if self.matches.len() > 1 {
            // If we have a regex set, then we have more than one ending
            // state, so leading to one of those states is generally
            // meaningless.
            return false;
        }
        match self[self.skip(pc)] {
            Inst::Match(_) => true,
            _ => false,
        }
    }

    /// Returns true if the current configuration demands that an implicit
    /// `.*?` be prepended to the instruction sequence.
    pub fn needs_dotstar(&self) -> bool {
        self.is_dfa && !self.is_reverse && !self.is_anchored_start
    }

    /// Returns true if this program uses Byte instructions instead of
    /// Char/Range instructions.
    pub fn uses_bytes(&self) -> bool {
        self.is_bytes || self.is_dfa
    }

    /// Returns true if this program exclusively matches valid UTF-8 bytes.
    ///
    /// That is, if an invalid UTF-8 byte is seen, then no match is possible.
    pub fn only_utf8(&self) -> bool {
        self.only_utf8
    }

    /// Return the approximate heap usage of this instruction sequence in
    /// bytes.
    pub fn approximate_size(&self) -> usize {
        // The only instruction that uses heap space is Ranges (for
        // Unicode codepoint programs) to store non-overlapping codepoint
        // ranges. To keep this operation constant time, we ignore them.
        (self.len() * mem::size_of::<Inst>())
            + (self.matches.len() * mem::size_of::<InstPtr>())
            + (self.captures.len() * mem::size_of::<Option<String>>())
            + (self.capture_name_idx.len()
                * (mem::size_of::<String>() + mem::size_of::<usize>()))
            + (self.byte_classes.len() * mem::size_of::<u8>())
            + self.prefixes.approximate_size()
    }
}

impl Deref for Program {
    type Target = [Inst];

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn deref(&self) -> &Self::Target {
        &*self.insts
    }
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Inst::*;

        fn with_goto(cur: usize, goto: usize, fmtd: String) -> String {
            if goto == cur + 1 {
                fmtd
            } else {
                format!("{} (goto: {})", fmtd, goto)
            }
        }

        fn visible_byte(b: u8) -> String {
            use std::ascii::escape_default;
            let escaped = escape_default(b).collect::<Vec<u8>>();
            String::from_utf8_lossy(&escaped).into_owned()
        }

        for (pc, inst) in self.iter().enumerate() {
            match *inst {
                Match(slot) => write!(f, "{:04} Match({:?})", pc, slot)?,
                Save(ref inst) => {
                    let s = format!("{:04} Save({})", pc, inst.slot);
                    write!(f, "{}", with_goto(pc, inst.goto, s))?;
                }
                Split(ref inst) => {
                    write!(
                        f,
                        "{:04} Split({}, {})",
                        pc, inst.goto1, inst.goto2
                    )?;
                }
                EmptyLook(ref inst) => {
                    let s = format!("{:?}", inst.look);
                    write!(f, "{:04} {}", pc, with_goto(pc, inst.goto, s))?;
                }
                Char(ref inst) => {
                    let s = format!("{:?}", inst.c);
                    write!(f, "{:04} {}", pc, with_goto(pc, inst.goto, s))?;
                }
                Ranges(ref inst) => {
                    let ranges = inst
                        .ranges
                        .iter()
                        .map(|r| format!("{:?}-{:?}", r.0, r.1))
                        .collect::<Vec<String>>()
                        .join(", ");
                    write!(
                        f,
                        "{:04} {}",
                        pc,
                        with_goto(pc, inst.goto, ranges)
                    )?;
                }
                Bytes(ref inst) => {
                    let s = format!(
                        "Bytes({}, {})",
                        visible_byte(inst.start),
                        visible_byte(inst.end)
                    );
                    write!(f, "{:04} {}", pc, with_goto(pc, inst.goto, s))?;
                }
            }
            if pc == self.start {
                write!(f, " (start)")?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a Program {
    type Item = &'a Inst;
    type IntoIter = slice::Iter<'a, Inst>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Inst is an instruction code in a Regex program.
///
/// Regrettably, a regex program either contains Unicode codepoint
/// instructions (Char and Ranges) or it contains byte instructions (Bytes).
/// A regex program can never contain both.
///
/// It would be worth investigating splitting this into two distinct types and
/// then figuring out how to make the matching engines polymorphic over those
/// types without sacrificing performance.
///
/// Other than the benefit of moving invariants into the type system, another
/// benefit is the decreased size. If we remove the `Char` and `Ranges`
/// instructions from the `Inst` enum, then its size shrinks from 40 bytes to
/// 24 bytes. (This is because of the removal of a `Vec` in the `Ranges`
/// variant.) Given that byte based machines are typically much bigger than
/// their Unicode analogues (because they can decode UTF-8 directly), this ends
/// up being a pretty significant savings.
#[derive(Clone, Debug)]
pub enum Inst {
    /// Match indicates that the program has reached a match state.
    ///
    /// The number in the match corresponds to the Nth logical regular
    /// expression in this program. This index is always 0 for normal regex
    /// programs. Values greater than 0 appear when compiling regex sets, and
    /// each match instruction gets its own unique value. The value corresponds
    /// to the Nth regex in the set.
    Match(usize),
    /// Save causes the program to save the current location of the input in
    /// the slot indicated by InstSave.
    Save(InstSave),
    /// Split causes the program to diverge to one of two paths in the
    /// program, preferring goto1 in InstSplit.
    Split(InstSplit),
    /// EmptyLook represents a zero-width assertion in a regex program. A
    /// zero-width assertion does not consume any of the input text.
    EmptyLook(InstEmptyLook),
    /// Char requires the regex program to match the character in InstChar at
    /// the current position in the input.
    Char(InstChar),
    /// Ranges requires the regex program to match the character at the current
    /// position in the input with one of the ranges specified in InstRanges.
    Ranges(InstRanges),
    /// Bytes is like Ranges, except it expresses a single byte range. It is
    /// used in conjunction with Split instructions to implement multi-byte
    /// character classes.
    Bytes(InstBytes),
}

impl Inst {
    /// Returns true if and only if this is a match instruction.
    pub fn is_match(&self) -> bool {
        match *self {
            Inst::Match(_) => true,
            _ => false,
        }
    }
}

/// Representation of the Save instruction.
#[derive(Clone, Debug)]
pub struct InstSave {
    /// The next location to execute in the program.
    pub goto: InstPtr,
    /// The capture slot (there are two slots for every capture in a regex,
    /// including the zeroth capture for the entire match).
    pub slot: usize,
}

/// Representation of the Split instruction.
#[derive(Clone, Debug)]
pub struct InstSplit {
    /// The first instruction to try. A match resulting from following goto1
    /// has precedence over a match resulting from following goto2.
    pub goto1: InstPtr,
    /// The second instruction to try. A match resulting from following goto1
    /// has precedence over a match resulting from following goto2.
    pub goto2: InstPtr,
}

/// Representation of the `EmptyLook` instruction.
#[derive(Clone, Debug)]
pub struct InstEmptyLook {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The type of zero-width assertion to check.
    pub look: EmptyLook,
}

/// The set of zero-width match instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyLook {
    /// Start of line or input.
    StartLine,
    /// End of line or input.
    EndLine,
    /// Start of input.
    StartText,
    /// End of input.
    EndText,
    /// Word character on one side and non-word character on other.
    WordBoundary,
    /// Word character on both sides or non-word character on both sides.
    NotWordBoundary,
    /// ASCII word boundary.
    WordBoundaryAscii,
    /// Not ASCII word boundary.
    NotWordBoundaryAscii,
}

/// Representation of the Char instruction.
#[derive(Clone, Debug)]
pub struct InstChar {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The character to test.
    pub c: char,
}

/// Representation of the Ranges instruction.
#[derive(Clone, Debug)]
pub struct InstRanges {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The set of Unicode scalar value ranges to test.
    pub ranges: Vec<(char, char)>,
}

impl InstRanges {
    /// Tests whether the given input character matches this instruction.
    pub fn matches(&self, c: Char) -> bool {
        // This speeds up the `match_class_unicode` benchmark by checking
        // some common cases quickly without binary search. e.g., Matching
        // a Unicode class on predominantly ASCII text.
        for r in self.ranges.iter().take(4) {
            if c < r.0 {
                return false;
            }
            if c <= r.1 {
                return true;
            }
        }
        self.ranges
            .binary_search_by(|r| {
                if r.1 < c {
                    Ordering::Less
                } else if r.0 > c {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok()
    }

    /// Return the number of distinct characters represented by all of the
    /// ranges.
    pub fn num_chars(&self) -> usize {
        self.ranges
            .iter()
            .map(|&(s, e)| 1 + (e as u32) - (s as u32))
            .sum::<u32>() as usize
    }
}

/// Representation of the Bytes instruction.
#[derive(Clone, Debug)]
pub struct InstBytes {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The start (inclusive) of this byte range.
    pub start: u8,
    /// The end (inclusive) of this byte range.
    pub end: u8,
}

impl InstBytes {
    /// Returns true if and only if the given byte is in this range.
    pub fn matches(&self, byte: u8) -> bool {
        self.start <= byte && byte <= self.end
    }
}
use s3::creds::Credentials;
use s3::region::Region;

pub mod files;
pub mod inode;
pub mod inodetree;
pub mod datasource;

pub struct Storage {
    pub name: String,
    pub region: Region,
    pub credentials: Credentials,
    pub bucket: String,
}

impl Storage {
    pub fn new(name:String, region:Region,bucket:String) -> Self{
        Self{
            name,
            region,
            credentials: Credentials::default_blocking().unwrap(),
            bucket,
        }
    }
}
use crate::connector::Connector;

fn create_connector()-> Connector{
    let user = "rust".to_string();
    let host = "localhost".to_string();
    let database = "rust_test".to_string();
    Connector::new(user, host, database)
}

#[test]
pub fn test_connector() {
    let connector = create_connector();
    let mut client = connector.client().unwrap();
    let err = client.batch_execute("
        CREATE TABLE IF NOT EXISTS test (
            id              SERIAL PRIMARY KEY,
            data            VARCHAR NOT NULL
            )
    ").unwrap();
}

#[cfg(test)]
mod pg_connection{
    use crate::connection::PgConnection;
    use crate::connection::Connection;
    use crate::models::Model;

    #[test]
    pub fn test_pg_connection(){
        let client = super::create_connector().client().unwrap();
        let pg_connection = PgConnection::new(client);
    }

    struct TestModel{
        id:u32,
    }
    impl Model for TestModel{
    }
    #[test]
    pub fn test_pg_create_model(){
        let client = super::create_connector().client().unwrap();
        let pg_connection = PgConnection::new(client);
        pg_connection.register_model(TestModel {id:10});
    }
}
// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ptr;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;
use std::thread::{self, ThreadId};

use crate::error::{Error, Result};
use crate::grpc_sys::{self, gpr_clock_type, grpc_completion_queue};
use crate::task::UnfinishedWork;

pub use crate::grpc_sys::grpc_completion_type as EventType;
pub use crate::grpc_sys::grpc_event as Event;

/// `CompletionQueueHandle` enable notification of the completion of asynchronous actions.
pub struct CompletionQueueHandle {
    cq: *mut grpc_completion_queue,
    // When `ref_cnt` < 0, a shutdown is pending, completion queue should not
    // accept requests anymore; when `ref_cnt` == 0, completion queue should
    // be shutdown; When `ref_cnt` > 0, completion queue can accept requests
    // and should not be shutdown.
    ref_cnt: AtomicIsize,
}

unsafe impl Sync for CompletionQueueHandle {}
unsafe impl Send for CompletionQueueHandle {}

impl CompletionQueueHandle {
    pub fn new() -> CompletionQueueHandle {
        CompletionQueueHandle {
            cq: unsafe { grpc_sys::grpc_completion_queue_create_for_next(ptr::null_mut()) },
            ref_cnt: AtomicIsize::new(1),
        }
    }

    fn add_ref(&self) -> Result<()> {
        loop {
            let cnt = self.ref_cnt.load(Ordering::SeqCst);
            if cnt <= 0 {
                // `shutdown` has been called, reject any requests.
                return Err(Error::QueueShutdown);
            }
            let new_cnt = cnt + 1;
            if cnt
                == self
                    .ref_cnt
                    .compare_and_swap(cnt, new_cnt, Ordering::SeqCst)
            {
                return Ok(());
            }
        }
    }

    fn unref(&self) {
        let shutdown = loop {
            let cnt = self.ref_cnt.load(Ordering::SeqCst);
            // If `shutdown` is not called, `cnt` > 0, so minus 1 to unref.
            // If `shutdown` is called, `cnt` < 0, so plus 1 to unref.
            let new_cnt = cnt - cnt.signum();
            if cnt
                == self
                    .ref_cnt
                    .compare_and_swap(cnt, new_cnt, Ordering::SeqCst)
            {
                break new_cnt == 0;
            }
        };
        if shutdown {
            unsafe {
                grpc_sys::grpc_completion_queue_shutdown(self.cq);
            }
        }
    }

    fn shutdown(&self) {
        let shutdown = loop {
            let cnt = self.ref_cnt.load(Ordering::SeqCst);
            if cnt <= 0 {
                // `shutdown` is called, skipped.
                return;
            }
            // Make cnt negative to indicate that `shutdown` has been called.
            // Because `cnt` is initialized to 1, so minus 1 to make it reach
            // toward 0. That is `new_cnt = -(cnt - 1) = -cnt + 1`.
            let new_cnt = -cnt + 1;
            if cnt
                == self
                    .ref_cnt
                    .compare_and_swap(cnt, new_cnt, Ordering::SeqCst)
            {
                break new_cnt == 0;
            }
        };
        if shutdown {
            unsafe {
                grpc_sys::grpc_completion_queue_shutdown(self.cq);
            }
        }
    }
}

impl Drop for CompletionQueueHandle {
    fn drop(&mut self) {
        unsafe { grpc_sys::grpc_completion_queue_destroy(self.cq) }
    }
}

pub struct CompletionQueueRef<'a> {
    queue: &'a CompletionQueue,
}

impl<'a> CompletionQueueRef<'a> {
    pub fn as_ptr(&self) -> *mut grpc_completion_queue {
        self.queue.handle.cq
    }
}

impl<'a> Drop for CompletionQueueRef<'a> {
    fn drop(&mut self) {
        self.queue.handle.unref();
    }
}

/// `WorkQueue` stores the unfinished work of a completion queue.
///
/// Every completion queue has a work queue, and every work queue belongs
/// to exact one completion queue. `WorkQueue` is a short path for future
/// notifications. When a future is ready to be polled, there are two way
/// to notify it.
/// 1. If it's in the same thread where the future is spawned, the future
///    will be pushed into `WorkQueue` and be polled when current call tag
///    is handled;
/// 2. If not, the future will be wrapped as a call tag and pushed into
///    completion queue and finally popped at the call to `grpc_completion_queue_next`.
pub struct WorkQueue {
    id: ThreadId,
    pending_work: UnsafeCell<VecDeque<UnfinishedWork>>,
}

unsafe impl Sync for WorkQueue {}
unsafe impl Send for WorkQueue {}

const QUEUE_CAPACITY: usize = 4096;

impl WorkQueue {
    pub fn new() -> WorkQueue {
        WorkQueue {
            id: std::thread::current().id(),
            pending_work: UnsafeCell::new(VecDeque::with_capacity(QUEUE_CAPACITY)),
        }
    }

    /// Pushes an unfinished work into the inner queue.
    ///
    /// If the method is not called from the same thread where it's created,
    /// the work will returned and no work is pushed.
    pub fn push_work(&self, work: UnfinishedWork) -> Option<UnfinishedWork> {
        if self.id == thread::current().id() {
            unsafe { &mut *self.pending_work.get() }.push_back(work);
            None
        } else {
            Some(work)
        }
    }

    /// Pops one unfinished work.
    ///
    /// It should only be called from the same thread where the queue is created.
    /// Otherwise it leads to undefined behavior.
    pub unsafe fn pop_work(&self) -> Option<UnfinishedWork> {
        let queue = &mut *self.pending_work.get();
        if queue.capacity() > QUEUE_CAPACITY && queue.len() < queue.capacity() / 2 {
            queue.shrink_to_fit();
        }
        { &mut *self.pending_work.get() }.pop_back()
    }
}

#[derive(Clone)]
pub struct CompletionQueue {
    handle: Arc<CompletionQueueHandle>,
    pub(crate) worker: Arc<WorkQueue>,
}

impl CompletionQueue {
    pub fn new(handle: Arc<CompletionQueueHandle>, worker: Arc<WorkQueue>) -> CompletionQueue {
        CompletionQueue { handle, worker }
    }

    /// Blocks until an event is available, the completion queue is being shut down.
    pub fn next(&self) -> Event {
        unsafe {
            let inf = grpc_sys::gpr_inf_future(gpr_clock_type::GPR_CLOCK_REALTIME);
            grpc_sys::grpc_completion_queue_next(self.handle.cq, inf, ptr::null_mut())
        }
    }

    pub fn borrow(&self) -> Result<CompletionQueueRef<'_>> {
        self.handle.add_ref()?;
        Ok(CompletionQueueRef { queue: self })
    }

    /// Begin destruction of a completion queue.
    ///
    /// Once all possible events are drained then `next()` will start to produce
    /// `Event::QueueShutdown` events only.
    pub fn shutdown(&self) {
        self.handle.shutdown()
    }

    pub fn worker_id(&self) -> ThreadId {
        self.worker.id
    }
}
//! On-board user LEDs
//!
//! - Red = Pin 22
//! - Green = Pin 19
//! - Blue = Pin 21
use embedded_hal::digital::v2::OutputPin;
use e310x_hal::gpio::gpio0::{Pin19, Pin21, Pin22};
use e310x_hal::gpio::{Output, Regular, Invert};

/// Red LED
pub type RED = Pin22<Output<Regular<Invert>>>;

/// Green LED
pub type GREEN = Pin19<Output<Regular<Invert>>>;

/// Blue LED
pub type BLUE = Pin21<Output<Regular<Invert>>>;

/// Returns RED, GREEN and BLUE LEDs.
pub fn rgb<X, Y, Z>(
    red: Pin22<X>, green: Pin19<Y>, blue: Pin21<Z>
) -> (RED, GREEN, BLUE)
{
    let red: RED = red.into_inverted_output();
    let green: GREEN = green.into_inverted_output();
    let blue: BLUE = blue.into_inverted_output();
    (red, green, blue)
}

/// Generic LED
pub trait Led {
    /// Turns the LED off
    fn off(&mut self);

    /// Turns the LED on
    fn on(&mut self);
}

impl Led for RED {
    fn off(&mut self) {
        self.set_low().unwrap();
    }

    fn on(&mut self) {
        self.set_high().unwrap();
    }
}

impl Led for GREEN {
    fn off(&mut self) {
        self.set_low().unwrap();
    }

    fn on(&mut self) {
        self.set_high().unwrap();
    }
}

impl Led for BLUE {
    fn off(&mut self) {
        self.set_low().unwrap();
    }

    fn on(&mut self) {
        self.set_high().unwrap();
    }
}
//! Board support crate for HiFive1 and LoFive boards

#![deny(missing_docs)]
#![no_std]

pub use e310x_hal as hal;

pub mod clock;
pub use clock::configure as configure_clocks;

pub mod flash;

#[cfg(any(feature = "board-hifive1", feature = "board-hifive1-revb"))]
pub mod led;
#[cfg(any(feature = "board-hifive1", feature = "board-hifive1-revb"))]
pub use led::{RED, GREEN, BLUE, rgb, Led};

pub mod stdout;
pub use stdout::configure as configure_stdout;

#[doc(hidden)]
pub mod gpio;
use s3::bucket::Bucket;
use s3::creds::Credentials;
use s3::region::Region;
use std::path::Path;

use freefs::{
    datasource::sources::s3_bucket::BucketSource, files::Files, Storage,
};
use keys_grpc_rs::KeysManager;

fn main() {
    let bb = Storage {
        name: "backblaze".to_string(),
        region: Region::Custom {
            region: "us-west-002".to_string(),
            endpoint: "https://s3.us-west-002.backblazeb2.com".to_string(),
        },
        credentials: Credentials::default_blocking().unwrap(),
        bucket: "rust-test".to_string(),
    };
    let mut manager = KeysManager::new("freefs".to_string());
    //manager.auth_setup();
    manager.auth_unlock();
    let bucket = create_bucket(bb);
    let files = bucket.list_blocking("test-folder".to_string(), None).unwrap();
    for (i, (file,code)) in files.iter().enumerate(){
        assert_eq!(&200, code);
        let contents = &file.contents;
        for obj in contents{
            println!("{:?}",obj.key);
        }
    }
    let mountpoint = Path::new("/tmp/rust-test");
    let data_source = BucketSource::new(bucket,"/tmp/rust-test-transient".to_string(),"/tmp/rust-test-stage".to_string(),manager);
    let fs = Files::new(data_source);
    mount(mountpoint,fs);
}
fn mount(mountpoint: &Path, fs: Files) {
    let result = fuse::mount(fs, &mountpoint, &[]).unwrap();
    println!("{:?}", result);
}

fn create_bucket(storage: Storage) -> Bucket {
    Bucket::new(&storage.bucket, storage.region, storage.credentials).unwrap()
}
pub trait Model{
    fn test(&self){
        println!("Model trait!");
    }
}
use crate::{
    datasource::{sources::s3_bucket::BucketSource, DataSource},
    inode::INode,
    inodetree::INodeTree,
};
use fuse::{
    FileType, Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyWrite, Request,
};
use futures::executor::block_on;
use libc::{EINVAL, ENOENT, ENOTEMPTY, EROFS};
use std::ffi::OsStr;
use std::path::Path;
use std::{
    sync::Arc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};


pub struct Files {
    data_source: Arc<BucketSource>,
    inode_tree: INodeTree,
}

impl Files {
    pub fn new(data_source: BucketSource) -> Self {
        let mut new = Files {
            data_source: Arc::new(data_source),
            inode_tree: INodeTree::new(),
        };
        new.inode_tree.add(INode::new(
            "",
            1,
            0,
            UNIX_EPOCH,
            FileType::Directory,
            0,
            blake3::hash(b""),
        ));
        new.load_from_inode_tree_log();
        for mut node in &mut new.inode_tree.nodes {
            if node.kind != FileType::Directory {
                let (size, mtime) = new
                    .data_source
                    .get_data_attr(&node.hash)
                    .unwrap_or((0, UNIX_EPOCH));
                node.size = size;
                node.mtime = mtime;
            }
        }
        let user = new
            .data_source
            .get_manager_user()
            .expect("Could not find a key manager user");
        println!("User: {:?}", user);
        new.inode_tree.add_inode_dir(&user, Some(1));
        new
    }

    fn write_inode_tree(&self) {
        let log = self.inode_tree.write_all_to_string();
        self.data_source.put_log(log);
    }

    fn load_from_inode_tree_log(&mut self) {
        let nodes = self.data_source.get_log();
        self.inode_tree.add_from_keys(nodes, None);
    }

    fn sync_path(&self) {
        let data_source = Arc::clone(&self.data_source);
        let hashes = self.inode_tree.get_hash_list();
        thread::spawn(move || {
            match block_on(data_source.sync_stage_area(hashes)) {
                Err(e) => {
                    println!("Sync ERROR: {}", e);
                }
                _ => {}
            };
        });
        self.write_inode_tree();
    }

    fn folder_name_recipient(&self, name: &str) -> Option<Vec<String>> {
        if !name.contains(
            &self.data_source
                .get_manager_user()
                .expect("Could not find a key manager user"),
        ) {
            return None
        }

        let recipients: Vec<String> = keys_grpc_rs::get_users_from_string(name);
        match recipients.len() {
            0 => None,
            _ => Some(recipients),
        }
    }
}

impl Filesystem for Files {
    fn flush(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        println!("flush {}", ino);
        self.sync_path();
        println!("{}", self.inode_tree.write_all_to_string());
        reply.ok();
    }

    fn fsync(&mut self, _req: &Request<'_>, ino: u64, _fh: u64, datasync: bool, reply: ReplyEmpty) {
        println!("fsync {},{}", ino, datasync);
        reply.ok();
    }

    fn lookup(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        println!("lookup: {:?}, {}", name.to_str(), parent);

        let entry = match self
            .inode_tree
            .get_inode_from_key_parent(name.to_str().unwrap(), parent)
        {
            Ok(entry) => entry,
            Err(e) => {
                //println!("lookup ERROR: {}", e);
                reply.error(ENOENT);
                return;
            }
        };
        reply.entry(&Duration::from_secs(1), &entry.get_file_attr(), 0);
    }

    fn getattr(&mut self, _req: &Request, ino: u64, reply: ReplyAttr) {
        println!("getattr(ino={})", ino);
        let attr = match self.inode_tree.get_inode_from_ino(ino) {
            Ok(entry) => entry.get_file_attr(),
            Err(e) => {
                println!("gettattr ERROR");
                reply.error(ENOENT);
                return;
            }
        };
        let ttl = Duration::from_secs(1);
        reply.attr(&ttl, &attr);
    }

    fn setattr(
        &mut self,
        _req: &Request,
        ino: u64,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        _size: Option<u64>,
        _atime: Option<SystemTime>,
        _mtime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        println!("Set attr{:?}", ino);
        let attr = match self.inode_tree.get_inode_from_ino(ino) {
            Ok(entry) => entry.get_file_attr(),
            Err(e) => {
                println!("setattr ERROR");
                reply.error(ENOENT);
                return;
            }
        };
        reply.attr(&Duration::from_secs(1), &attr);
    }

    fn read(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        reply: ReplyData,
    ) {
        let end = (offset + size as i64) as usize;
        println!("read {:?} {:?}, {}", ino, offset, size);
        let hash = match self.inode_tree.get_inode_from_ino(ino) {
            Ok(entry) => entry.hash,
            Err(e) => {
                println!("setattr ERROR");
                reply.error(ENOENT);
                return;
            }
        };
        let mut blocks = match self.data_source.get_data(&hash) {
            Ok(result) => result,
            Err(e) => {
                println!("read ERROR: {}", e);
                reply.error(ENOENT);
                return;
            }
        };
        if blocks.len() < end {
            blocks.resize(end, 0);
        }
        let data = &blocks[offset as usize..end];
        reply.data(data);
    }

    fn readdir(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        println!("readdir: {:?}  \n offset: {:?}", ino, offset);

        let dir = match self.inode_tree.get_inode_from_ino(ino) {
            Ok(entry) => entry,
            Err(e) => {
                println!("gettattr ERROR");
                reply.error(ENOENT);
                return;
            }
        };

        let mut entries = vec![dir];
        entries.append(&mut self.inode_tree.get_children(dir.ino));

        for (i, entry) in entries.iter().enumerate().skip(offset as usize) {
            if i == 0 {
                reply.add(dir.ino, 1, FileType::Directory, &Path::new("."));
                reply.add(dir.ino, 2, FileType::Directory, &Path::new(".."));
                continue;
            }
            reply.add(entry.ino, i as i64 + 1, entry.kind, &Path::new(&entry.key));
            println!("{:?}: Adding {:?}, {:?}", i, entry.key, entry.ino);
        }
        reply.ok();
    }

    fn open(&mut self, _req: &Request, ino: u64, flags: u32, reply: ReplyOpen) {
        println!("open {:?}, {:?}", ino, flags);
        reply.opened(0, flags);
    }

    fn write(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _flags: u32,
        reply: ReplyWrite,
    ) {
        println!("Write: {:?}, {:?}, {:?} bytes", ino, offset, data.len());
        let hash;
        {
            let offset = offset as usize;
            let inode = match self.inode_tree.get_inode_from_ino(ino) {
                Ok(inode) => inode,
                Err(e) => {
                    println!("write ERROR: {}", e);
                    reply.error(ENOENT);
                    return;
                }
            };
            let mut blocks = match self.data_source.get_data(&inode.hash) {
                Ok(result) => result,
                Err(e) => {
                    println!("write ERROR:{}", e);
                    reply.error(ENOENT);
                    return;
                }
            };
            let buffer = data.len() + offset;
            if buffer > 0 {
                blocks.resize(buffer, 0);
            }
            blocks.splice(offset..(data.len() + offset), data.to_vec());
            let recipients = self.inode_tree.get_root_parent(ino).unwrap();
            let recipients = self.folder_name_recipient(&recipients.key);
            hash = self.data_source.put_data(blocks, recipients);
            match self.inode_tree.update_inode_hash(ino, hash.clone()) {
                Err(e) => println!("write ERROR:{}", e),
                _ => {}
            };
        }
        if let Ok((size, mtime)) = self.data_source.get_data_attr(&hash) {
            match self.inode_tree.update_inode_attr(ino, mtime, size) {
                Err(e) => println!("write ERROR:{}", e),
                _ => {}
            };
        }
        reply.written(data.len() as u32);
    }

    fn create(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        _mode: u32,
        _flags: u32,
        reply: ReplyCreate,
    ) {
        println!("Create: {:?},{:?}", name, parent);
        let name = name.to_str().unwrap();
        if parent == 1 {
            reply.error(EROFS);
            return;
        }

        let ino = self
            .inode_tree
            .add_empty(name.to_string(), blake3::hash(b""), parent);
        let recipients = self.inode_tree.get_root_parent(ino).unwrap();
        let recipients = self.folder_name_recipient(&recipients.key);
        let hash = self.data_source.put_data(vec![], recipients);
        match self.inode_tree.update_inode_hash(ino, hash) {
            Err(e) => println!("create ERROR:{}", e),
            _ => {}
        };

        let attr = match self.inode_tree.get_inode_from_ino(ino) {
            Ok(entry) => entry.get_file_attr(),
            Err(e) => {
                println!("create ERROR:{}", e);
                reply.error(ENOENT);
                return;
            }
        };

        reply.created(&Duration::from_secs(1), &attr, 0, 0, 0);
    }

    fn unlink(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        println!("Unlink {:?}, {:?}", name, parent);
        let ino: Option<u64>;
        {
            let entry = match self
                .inode_tree
                .get_inode_from_key_parent(name.to_str().unwrap(), parent)
            {
                Ok(entry) => entry,
                Err(e) => {
                    println!("unlink ERROR:{}", e);
                    reply.error(ENOENT);
                    return;
                }
            };
            match block_on(self.data_source.delete_data(&entry.hash)) {
                Err(e) => {
                    println!("unlink ERROR: {}", e);
                }
                _ => {}
            };
            ino = Some(entry.ino);
        }
        match ino {
            Some(ino) => self.inode_tree.remove(ino),
            None => {}
        }
        println!("{}", self.inode_tree.write_all_to_string());
        self.sync_path();

        reply.ok();
    }

    fn mkdir(&mut self, _req: &Request, parent: u64, name: &OsStr, _mode: u32, reply: ReplyEntry) {
        let name = name.to_str().unwrap();
        println!("mkdir {}, {}", parent, name);
        if parent == 1 {
            if let None = self.folder_name_recipient(name) {
                reply.error(EINVAL);
                return;
            }
        }
        let dir_ino = self.inode_tree.add_inode_dir(name, Some(parent));
        let attr = self
            .inode_tree
            .get_inode_from_ino(dir_ino)
            .unwrap()
            .get_file_attr();
        reply.entry(&Duration::from_secs(1), &attr, 0);
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name = name.to_str().unwrap();
        println!("rmdir {}, {}", parent, name);
        let dir = match self.inode_tree.get_inode_from_key_parent(name, parent) {
            Ok(entry) => entry.ino,
            Err(e) => {
                println!("rmdir ERROR: {}", e);
                reply.error(ENOENT);
                return;
            }
        };
        let children_count = self.inode_tree.get_children(dir).iter().count();
        if children_count > 0 {
            reply.error(ENOTEMPTY);
        } else {
            self.inode_tree.remove(dir);
            reply.ok();
        }
    }
    fn init(&mut self, _req: &Request<'_>) -> Result<(), libc::c_int> {
        Ok(())
    }
    fn destroy(&mut self, _req: &Request<'_>) {}
    fn forget(&mut self, _req: &Request<'_>, _ino: u64, _nlookup: u64) {}
    fn readlink(&mut self, _req: &Request<'_>, _ino: u64, reply: ReplyData) {
        reply.error(libc::ENOSYS);
    }
    fn mknod(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _mode: u32,
        _rdev: u32,
        reply: ReplyEntry,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn symlink(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _link: &Path,
        reply: ReplyEntry,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn rename(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _newparent: u64,
        _newname: &OsStr,
        reply: ReplyEmpty,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn link(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _newparent: u64,
        _newname: &OsStr,
        reply: ReplyEntry,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn release(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _flags: u32,
        _lock_owner: u64,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }
    fn opendir(&mut self, _req: &Request<'_>, _ino: u64, _flags: u32, reply: ReplyOpen) {
        reply.opened(0, 0);
    }
    fn releasedir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }
    fn fsyncdir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn statfs(&mut self, _req: &Request<'_>, _ino: u64, reply: fuse::ReplyStatfs) {
        reply.statfs(0, 0, 0, 0, 0, 512, 255, 0);
    }
    fn setxattr(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _name: &OsStr,
        _value: &[u8],
        _flags: u32,
        _position: u32,
        reply: ReplyEmpty,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn getxattr(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _name: &OsStr,
        _size: u32,
        reply: fuse::ReplyXattr,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn listxattr(&mut self, _req: &Request<'_>, _ino: u64, _size: u32, reply: fuse::ReplyXattr) {
        reply.error(libc::ENOSYS);
    }
    fn removexattr(&mut self, _req: &Request<'_>, _ino: u64, _name: &OsStr, reply: ReplyEmpty) {
        reply.error(libc::ENOSYS);
    }
    fn access(&mut self, _req: &Request<'_>, _ino: u64, _mask: u32, reply: ReplyEmpty) {
        reply.error(libc::ENOSYS);
    }
    fn getlk(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _lock_owner: u64,
        _start: u64,
        _end: u64,
        _typ: u32,
        _pid: u32,
        reply: fuse::ReplyLock,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn setlk(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _lock_owner: u64,
        _start: u64,
        _end: u64,
        _typ: u32,
        _pid: u32,
        _sleep: bool,
        reply: ReplyEmpty,
    ) {
        reply.error(libc::ENOSYS);
    }
    fn bmap(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _blocksize: u32,
        _idx: u64,
        reply: fuse::ReplyBmap,
    ) {
        reply.error(libc::ENOSYS);
    }
}
use fuse::{FileAttr, FileType};
use std::time::SystemTime;

#[derive(Clone)]
pub struct INode {
    pub key: String,
    pub ino: u64,
    pub size: u64,
    pub mtime: SystemTime,
    pub kind: FileType,
    pub parent: u64,
    pub hash:blake3::Hash,
}

impl INode {
    pub fn new(
        key: &str,
        ino: u64,
        size: u64,
        mtime: SystemTime,
        kind: FileType,
        parent: u64,
        hash: blake3::Hash,
    ) -> Self {
        INode {
            key: key.to_string(),
            ino,
            size,
            mtime,
            kind,
            parent,
            hash,
        }
    }

    pub fn get_file_attr(&self) -> FileAttr {
        FileAttr {
            ino: self.ino,
            size: self.size,
            blocks: (self.size * 10 / 4096) / 10,
            atime: self.mtime,
            mtime: self.mtime,
            ctime: self.mtime,
            crtime: self.mtime,
            kind: self.kind,
            perm: 0o755,
            nlink: 1,
            uid: 1,
            gid: 1,
            rdev: 0,
            flags: 0,
        }
    }
}
use std::{collections::VecDeque, time::Instant};

use cached::proc_macro::cached;
use image::{GenericImage, GenericImageView, Pixel, Rgb};
use photon_rs::{
    channels::{self, alter_blue_channel, alter_green_channel, alter_red_channel},
    conv::noise_reduction,
    helpers::dyn_image_from_raw,
    multiple::blend,
    native::{open_image, save_image},
    transform::{
        crop, padding_bottom, padding_left, padding_right, padding_top, padding_uniform, resize,
        SamplingFilter,
    },
    PhotonImage, Rgba,
};
use clap::Parser;

#[derive(Parser,Debug)]
#[clap(author, version, about, long_about = None)]
struct Args{
    file:String,
    learn:f64,
    epsilon:f64,
    scale:u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let vec = vec![0,0,0,0];
    let mut img = PhotonImage::new(vec,1,1);
    save_image(img, "t.jpg");
    let args = Args::parse();
    let img = open_image(&args.file).expect("File should open");
    //let img = open_image("data/00056v.jpg").expect("File should open");
    println!("w,h {} {}", img.get_width(), img.get_height());
    let orig = realign(img.clone(), Method::None, 10000.0, 1.0, 1);
    //let new = realign(img.clone(), Method::Scale, 10000.0, 100.0, args.scale);
    let new = realign(img, Method::GradientDescent, args.learn, args.epsilon, args.scale);
    save_image(orig, "img.jpg")?;
    save_image(new.clone(), "img2.jpg")?;
    //save_image(new, format!("processed/{}",args.file).as_str())?;
    //save_image(brute, "img3.jpg")?;
    Ok(())
}
enum Method {
    Scale,
    GradientDescent,
    None,
}

#[derive(Clone, Copy)]
enum Channel {
    R = 0,
    G = 1,
    B = 2,
}
#[derive(Clone, Copy)]
struct Channels {
    a: Channel,
    b: Channel,
}

impl Channels {
    fn new((a, b): (Channel, Channel)) -> Self {
        Self { a, b }
    }
}
impl From<(Channel, Channel)> for Channels {
    fn from(p: (Channel, Channel)) -> Self {
        Channels::new(p)
    }
}

fn realign(img: PhotonImage, method: Method, delta: f64, epsilon: f64, scale: u32) -> PhotonImage {
    let mut img_orig = img.clone();
    let mut img = resize(
        &img,
        img.get_width() / scale,
        img.get_height() / scale,
        SamplingFilter::Nearest,
    );
    let width = img.get_width();
    let height = img.get_height();
    let mut b_channel = crop(&mut img, 0, 0, width, height / 3);
    let mut g_channel = crop(&mut img, 0, height / 3, width, 2 * height / 3);
    let mut r_channel = crop(&mut img, 0, 2 * height / 3, width, 3 * height / 3);
    reverse_grayscale(&mut r_channel, 0);
    reverse_grayscale(&mut g_channel, 1);
    reverse_grayscale(&mut b_channel, 2);
    let (p_rg, p_gb,p_rb,p_rg_p, p_gb_p,p_rb_p) = match method {
        Method::Scale => {
            let (g, b, p_gb) = brute_force(g_channel, b_channel, (Channel::G, Channel::B).into(), 15,true);
            let (g, b, p_gb_p) = brute_force(g, b, (Channel::G, Channel::B).into(), 15,false);
            let (r, g, p_rg) =
                brute_force(r_channel, g, (Channel::R, Channel::G).into(), 15,true);

            let (r, g, p_rg_p) =
                brute_force(r, g, (Channel::R, Channel::G).into(), 15,false);
            let (r, b, p_rb) = brute_force(r, b, (Channel::R, Channel::B).into(), 15,true);
            let (_, _, p_rb_p) = brute_force(r, b, (Channel::R, Channel::B).into(), 15,false);
            (p_rg, p_gb,p_rb,p_rg_p, p_gb_p,p_rb_p)
        }
        Method::GradientDescent => {
            let (g, b, p_gb) = gradient_descent(
                g_channel,
                b_channel,
                (Channel::G, Channel::B).into(),
                delta,
                epsilon,
                true,
            );

            let (g, b, p_gb_p) = gradient_descent(
                g,
                b,
                (Channel::G, Channel::B).into(),
                delta,
                epsilon,
                false,
            );
            let (r, g, p_rg) = gradient_descent(
                r_channel,
                g,
                (Channel::R, Channel::G).into(),
                delta,
                epsilon,
                true
            );
            let (r, g, p_rg_p) =
                gradient_descent(r, g, (Channel::R, Channel::G).into(), delta, epsilon,false);

            let (r, b, p_rb) =
                gradient_descent(r, b, (Channel::R, Channel::B).into(), delta, epsilon,true);
            let (r, b, p_rb_p) =
                gradient_descent(r, b, (Channel::R, Channel::B).into(), delta, epsilon,false);
            (p_rg, p_gb,p_rb,p_rg_p, p_gb_p,p_rb_p)
        }
        Method::None => (0.0, 0.0,0.0,0.0, 0.0,0.0),
    };

    let width = img_orig.get_width();
    let height = img_orig.get_height();
    let p_rg = (p_rg * scale as f32) as i32;
    let p_gb = (p_gb * scale as f32) as i32;
    let p_rb = (p_rb * scale as f32) as i32;
    let p_rg_p = (-p_rg_p * scale as f32) as i32;
    let p_gb_p = (-p_gb_p * scale as f32) as i32;
    let p_rb_p = (-p_rb_p * scale as f32) as i32;
    let b = crop(&mut img_orig, 0, 0, width, height / 3);
    let g = crop(&mut img_orig, 0, height / 3, width, 2 * height / 3);
    let r = crop(&mut img_orig, 0, 2 * height / 3, width, 3 * height / 3);
    let (mut g, mut b) = pad_photo(p_gb_p, &g, &b,false);
    let (mut r, mut b) = pad_photo(p_rb, &r, &b,true);
    let (r, g) = pad_photo(p_rg, &r, &g,true);
    let (mut r, g) = pad_photo(p_rg_p, &r, &g,false);
    let (mut g, b) = pad_photo(p_gb, &g, &b,true);
    let (mut r, mut b) = pad_photo(p_rb_p, &r, &b,false);
    println!("rg [{},{}] rb [{},{}] gb [{},{}]",p_rg,&p_rg_p,&p_rb,p_rb_p,p_gb,p_gb_p);
    reverse_grayscale(&mut r, 0);
    reverse_grayscale(&mut g, 1);
    reverse_grayscale(&mut b, 2);
    let orig = overlay(r, b);
    overlay(orig, g)
}

fn component_overlay(pi_a: &PhotonImage, pi_b: &PhotonImage) -> PhotonImage {
    let mut img_a = dyn_image_from_raw(pi_a);
    let img_b = dyn_image_from_raw(pi_b);
    let img_a_pixels = img_a.clone();
    let pixels_a = img_a_pixels.pixels();
    let img_b_pixels = img_b;
    let pixels_b = img_b_pixels.pixels();
    pixels_a
        .into_iter()
        .zip(pixels_b.into_iter())
        .for_each(|(mut a, b)| {
            a.2.channels_mut()[0] =
                (a.2.channels_mut()[0] as u16 + b.2.channels()[0] as u16).clamp(0, 255) as u8;
            a.2.channels_mut()[1] =
                (a.2.channels_mut()[1] as u16 + b.2.channels()[1] as u16).clamp(0, 255) as u8;
            a.2.channels_mut()[2] =
                (a.2.channels_mut()[2] as u16 + b.2.channels()[2] as u16).clamp(0, 255) as u8;
            img_a.put_pixel(a.0, a.1, a.2);
        });
    let raw_pixels = img_a.to_bytes();
    PhotonImage::new(raw_pixels, pi_a.get_width(), pi_a.get_height())
}

fn overlay(mut pi_a: PhotonImage, mut pi_b: PhotonImage) -> PhotonImage {
    if pi_a.get_width() < pi_b.get_width() {
        std::mem::swap(&mut pi_a, &mut pi_b);
    }
    let b_x_pad = pi_a.get_width() - pi_b.get_width();
    let mut pi_b = padding_right(&pi_b, b_x_pad, Rgba::new(0, 0, 0, 0));

    if pi_a.get_height() < pi_b.get_height() {
        std::mem::swap(&mut pi_a, &mut pi_b);
    }
    let b_y_pad = pi_a.get_height() - pi_b.get_height();
    let pi_b = padding_bottom(&pi_b, b_y_pad, Rgba::new(0, 0, 0, 0));

    component_overlay(&pi_a, &pi_b)
}

fn reverse_grayscale(photon_image: &mut PhotonImage, channel: usize) {
    if channel != 0 {
        alter_red_channel(photon_image, -255);
    }
    if channel != 1 {
        alter_green_channel(photon_image, -255);
    }
    if channel != 2 {
        alter_blue_channel(photon_image, -255);
    }
}

// Should DP
fn difference(pi_a: &PhotonImage, pi_b: &PhotonImage) -> f64 {
    let pixels_a = dyn_image_from_raw(pi_a);
    let pixels_a = pixels_a.pixels();
    let pixels_b = dyn_image_from_raw(pi_b);
    let pixels_b = pixels_b.pixels();
    let (a, b, c) = pixels_a
        .into_iter()
        .zip(pixels_b.into_iter())
        .map(|((_, _, a), (_, _, b))| {
            if a.channels()[3] == 0 || b.channels()[3] == 0{
                return (0.0,0.0,0.0);
            }
            let b = b.to_luma();
            let a = a.to_luma();
            let mut b = b.channels()[0] as f64;
            let a = a.channels()[0] as f64;
            (a*a * b*b, a.powf(4.0), b.powf(4.0))
        })
        .fold((0.0, 0.0, 0.0), |(x, y, z), (a, b, c)| {
            (x + a, b + y, z + c)
        });
    a / (b.sqrt() * c.sqrt())
}

fn pad_photo(x: i32, pi_a: &PhotonImage, pi_b: &PhotonImage,side:bool) -> (PhotonImage, PhotonImage) {
    let (pi_a_y, pi_b_y) = if x > 0 { (x.abs(), 0) } else { (0, x.abs()) };
    if side{
        let pi_a_pad = padding_top(&pi_a, pi_a_y as u32, Rgba::new(0, 0, 0, 0));
        let pi_b_pad = padding_top(&pi_b, pi_b_y as u32, Rgba::new(0, 0, 0, 0));

        let pi_a_pad = padding_bottom(&pi_a_pad, pi_b_y as u32, Rgba::new(0, 0, 0, 0));
        let pi_b_pad = padding_bottom(&pi_b_pad, pi_a_y as u32, Rgba::new(0, 0, 0, 0));
        (pi_a_pad, pi_b_pad)
    }else{
      //let pi_a_pad = padding_left(&pi_a, pi_a_y as u32, Rgba::new(0, 0, 0, 0));
      //let pi_b_pad = padding_left(&pi_b, pi_b_y as u32, Rgba::new(0, 0, 0, 0));

      //let pi_a_pad = padding_right(&pi_a_pad, pi_b_y as u32, Rgba::new(0, 0, 0, 0));
      //let pi_b_pad = padding_right(&pi_b_pad, pi_a_y as u32, Rgba::new(0, 0, 0, 0));
        (pi_a.clone(), pi_b.clone())
    }
}

#[cached(
    size=20,
    key = "(i32,u8,u8,bool)",
    convert = r#"{(y,channels.a as u8,channels.b as u8,side)}"#
)]
fn pad_and_diff(
    y: i32,
    channels: Channels,
    pi_a: &PhotonImage,
    pi_b: &PhotonImage,
    side:bool,
) -> (PhotonImage, PhotonImage, f64) {
    let (pi_a_pad, pi_b_pad) = pad_photo(y, pi_a, pi_b,true);
    let diff = difference(&pi_a_pad, &pi_b_pad);
    (pi_a_pad, pi_b_pad, diff)
}

fn brute_force(
    pi_a: PhotonImage,
    pi_b: PhotonImage,
    channels: Channels,
    search_radius: i32,
    side:bool,
) -> (PhotonImage, PhotonImage, f32) {
    let mut best = 0.0;
    let mut best_a = pi_a.clone();
    let mut best_b = pi_b.clone();
    let mut best_p = 0.0;
    for y in (-search_radius)..search_radius {
        let (pi_a_pad, pi_b_pad, diff) = pad_and_diff(y, channels, &pi_a, &pi_b,side);
        //println!("y:{} diff: {}",y,diff);
        if diff > best {
            best = diff;
            best_a = pi_a_pad;
            best_b = pi_b_pad;
            best_p = y as f32;
        }
    }
    println!("_-----------------");
    (best_a, best_b, best_p)
}

fn gradient_descent(
    pi_a: PhotonImage,
    pi_b: PhotonImage,
    channels: Channels,
    delta: f64,
    mut epsilon: f64,
    side:bool
) -> (PhotonImage, PhotonImage, f32) {
    let mut padding = 0.0;
    let mut seen: VecDeque<i32> = VecDeque::new();
    let mut seen_cnt = 0;
    let mut delta = delta;
    let mut best = 1.0;
    let mut best_pad = 0.0;
    let mut pi_a_pad_final = pi_a.clone();
    let mut pi_b_pad_final = pi_b.clone();
    for _ in 1..300 {
        let (pi_a_pad, pi_b_pad, diff) = pad_and_diff((padding) as i32, channels, &pi_a, &pi_b,side);
        let (_, _, diff_p) = pad_and_diff((padding + epsilon) as i32, channels, &pi_a, &pi_b,side);

        let diff = 1.0 - diff;
        let diff_p = 1.0 - diff_p;
        let mut gradient = delta * ((diff - diff_p) / epsilon);
        let sign = gradient / gradient.abs();

        if gradient.abs() >= 1.0 {
            gradient = gradient.abs().log2() * sign;
        }
        gradient = (gradient * 100.0).trunc() / 100.0;
        if seen.contains(&(gradient as i32)) {
            seen_cnt += 1;
        } else {
            seen_cnt = 0;
        }
        seen.push_back(gradient as i32);
        if seen.len() > 10 {
            seen.pop_front();
        }
        println!(
            "x: {} {:.5} {:.5} {:.5} {} {:?}",
            gradient, padding, diff, diff_p, delta, epsilon
        );
        padding += gradient;
        if seen_cnt > 4 {
            delta /= 10.0;
        }
        if epsilon > gradient {
            epsilon/=1.1;
        }
        if diff < best {
            best = diff;
            pi_a_pad_final = pi_a_pad;
            pi_b_pad_final = pi_b_pad;
            best_pad = padding
        }
        if gradient.abs() < 0.01 {
            println!("-------");
            return (pi_a_pad_final, pi_b_pad_final, best_pad as f32);
        }
    }
    (pi_a_pad_final, pi_b_pad_final, best_pad as f32)
}
use fft2d::slice::fft_2d;
use image::{
    imageops::filter3x3, io::Reader as ImageReader, DynamicImage, GenericImage, GenericImageView,
    ImageBuffer, Pixel, RgbaImage, GrayImage,
};
use imageproc::{
    definitions::Image, drawing::draw_text_mut, filter::gaussian_blur_f32, map::map_colors2,
};
use rustfft::num_complex::Complex;
use rusttype::{Font, Scale};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
    a_blur: Option<f32>,
    b_blur: Option<f32>,
    c_blur: Option<f32>,
}

#[derive(Subcommand)]
enum Commands {
    File {
        file_a: String,
        file_b: String,
    },
    Text {
        msg1: String,
        msg2: String,
        msg3: Option<String>,
    },
}

const IDENTITY_MINUS_LAPLACIAN: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
const TEXT_COLOR_R: image::Rgba<u8> = image::Rgba([255, 0, 0, 255]);
const TEXT_COLOR_B: image::Rgba<u8> = image::Rgba([0, 0, 255, 255]);
const TEXT_COLOR_G: image::Rgba<u8> = image::Rgba([0, 255, 0, 255]);
const TEXT_COLOR_W: image::Rgba<u8> = image::Rgba([255, 255, 255, 255]);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let (img1, img2, img3) = match &args.command {
        Commands::File { file_a, file_b } => {
            let img1 = ImageReader::open(file_a)?.decode()?;
            let img2 = ImageReader::open(file_b)?.decode()?;
            (img1, img2, None)
        }
        Commands::Text { msg1, msg2, msg3 } => {
            let msg3_len = if let Some(msg) = msg3 { msg.len() } else { 0 };
            let len = msg1.len().max(msg2.len()).max(msg3_len);
            let width = ((400 * len) / 4) as u32;

            let img1 = draw_message(
                msg1.to_string(),
                width,
                200,
                20,
                35,
                Scale::uniform(150.0),
                TEXT_COLOR_R,
            );
            let img2 = draw_message(
                msg2.to_string(),
                width,
                200,
                20,
                35,
                Scale::uniform(150.0),
                TEXT_COLOR_G,
            );
            let img3 = if let Some(msg3) = msg3 {
                Some(draw_message(
                    msg3.to_string(),
                    width,
                    200,
                    20,
                    35,
                    Scale::uniform(150.0),
                    TEXT_COLOR_B,
                ))
            } else {
                None
            };
            (img1, img2, img3)
        }
    };

    create_fft(img1.clone()).save("fft_aa.jpg")?;
    create_fft(img2.clone()).save("fft_bb.jpg")?;
    img1.save("aa.jpg")?;
    img2.save("bb.jpg")?;
    let img1 = low_pass(img1, args.a_blur.unwrap_or(4.5));
    let img2 = high_pass(img2, args.b_blur.unwrap_or(0.545),args.a_blur.unwrap_or(4.5));
    img1.save("a.jpg")?;
    img2.save("b.jpg")?;

    create_fft(img1.clone()).save("fft_a.jpg")?;
    create_fft(img2.clone()).save("fft_b.jpg")?;
    let t = if let Some(img3) = img3 {
        img3.save("cc.jpg")?;

        create_fft(img3.clone()).save("fft_cc.jpg")?;
        let img3 = high_pass(img3, args.c_blur.unwrap_or(0.0),args.a_blur.unwrap_or(4.5));
        create_fft(img3.clone()).save("fft_c.jpg")?;
        img3.save("c.jpg")?;
        overlay3(img1, img2, img3)
    } else {
        overlay(img1, img2)
    };
    t.save("t.jpg")?;
    create_fft(t).save("fft_t.jpg")?;
    Ok(())
}

fn create_fft(img:DynamicImage)->GrayImage{
    let (h,w) = (img.height(),img.width());
    let mut buff: Vec<Complex<f64>> = img
        .into_luma8()
        .as_raw()
        .iter()
        .map(|&pix| Complex::new(pix as f64 / 255.0, 0.0))
        .collect();
    fft_2d(w as usize, h as usize,&mut buff);
    view_fft_norm(w, h, &buff)
}

fn view_fft_norm(width: u32, height: u32, img_buffer: &[Complex<f64>]) -> GrayImage {
    let fft_log_norm: Vec<f64> = img_buffer.iter().map(|c| c.norm().ln()).collect();
    let max_norm = fft_log_norm.iter().cloned().fold(0.0, f64::max);
    let fft_norm_u8: Vec<u8> = fft_log_norm
        .into_iter()
        .map(|x| ((x / max_norm) * 255.0) as u8)
        .collect();
    GrayImage::from_raw(width, height, fft_norm_u8).unwrap()
}

fn draw_message(
    msg: String,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    scale: Scale,
    color: image::Rgba<u8>,
) -> DynamicImage {
    let font_data: &[u8] = include_bytes!("/usr/share/fonts/FuturaLT-Bold.ttf");
    let font: Font<'static> = Font::try_from_bytes(font_data).unwrap();
    let canvas: RgbaImage = ImageBuffer::new(width, height);
    let mut img = DynamicImage::ImageRgba8(canvas);
    draw_text_mut(&mut img, color, x, y, scale, &font, &msg);
    img
}

fn clamp_sub(a: u8, b: u8, max: u8) -> u8 {
    if a < b {
        0
    } else {
        max.min(a - b)
    }
}

fn clamp_add(a: u8, b: u8, max: u8) -> u8 {
    if (a as u16 + b as u16) > max.into() {
        max
    } else {
        a + b
    }
}

fn low_pass(img: DynamicImage, amt: f32) -> DynamicImage {
    DynamicImage::ImageRgba8(gaussian_blur_f32(&img.to_rgba8(), amt))
}

fn laplacian(amt: f32) -> [f32; 9] {
    let mut v = IDENTITY_MINUS_LAPLACIAN;
    v[4] *= amt;
    v
}

fn high_pass(img: DynamicImage, amt: f32,amt2:f32) -> DynamicImage {
    let img_impulse = filter3x3(&img, &laplacian(amt));
    let img_low = low_pass(img, amt2);
    let diff = map_colors2(&img_impulse, &img_low, |mut p, q| {
        p.apply2(&q, |c1, c2| clamp_sub(c1, c2, u8::MAX));
        p.0[3] = 255;
        p
    });
    DynamicImage::ImageRgba8(diff)
}

fn overlay(a: DynamicImage, b: DynamicImage) -> DynamicImage {
    let diff = map_colors2(&a, &b, |mut p, q| {
        p.apply2(&q, |c1, c2| (clamp_add(c1, c2, u8::MAX)));
        p.0[3] = 255;
        p
    });
    DynamicImage::ImageRgba8(diff)
}

fn overlay3(a: DynamicImage, b: DynamicImage, c: DynamicImage) -> DynamicImage {
    let diff = map_colors3(&a, &b, &c, |mut p, q, r| {
        assert_eq!(p.channels().len(), q.channels().len());
        assert_eq!(p.channels().len(), r.channels().len());
        for i in 0..p.channels().len() - 1 {
            p.channels_mut()[i] = clamp_add(
                clamp_add(p.channels()[i], q.channels()[i], u8::MAX),
                r.channels()[i],
                u8::MAX,
            );
        }
        p
    });
    DynamicImage::ImageRgba8(diff)
}

fn map_colors3<I, J, K, P, Q, R, S, F>(image1: &I, image2: &J, image3: &K, f: F) -> Image<S>
where
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = Q>,
    K: GenericImage<Pixel = R>,
    P: Pixel,
    Q: Pixel,
    R: Pixel,
    S: Pixel + 'static,
    F: Fn(P, Q, R) -> S,
{
    assert_eq!(image1.dimensions(), image2.dimensions());

    let (width, height) = image1.dimensions();
    let mut out: ImageBuffer<S, Vec<S::Subpixel>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let p = image1.unsafe_get_pixel(x, y);
                let q = image2.unsafe_get_pixel(x, y);
                let r = image3.unsafe_get_pixel(x, y);
                out.unsafe_put_pixel(x, y, f(p, q, r));
            }
        }
    }

    out
}
use serde::Deserialize;
use std::{fs, net::IpAddr, net::Ipv4Addr, net::SocketAddr};

#[derive(Deserialize)]
struct Config {
    server: Option<ServerConfig>,
}

#[derive(Deserialize)]
struct ServerConfig {
    ip_addr: Option<Ipv4Addr>,
    port: Option<u16>,
    tls_cert: Option<String>,
    tls_key: Option<String>,
}

fn main() {
    println!("cargo:rerun-if-changed=Settings.toml");
    let config = get_config();
    set_server(config.server);
}

fn get_config() -> Config {
    let config_str = match fs::read_to_string("Settings.toml") {
        Err(_) => panic!("Error, Settings.toml is missing!"),
        Ok(config_str) => config_str,
    };
    match toml::from_str(&config_str) {
        Err(err) => panic!("Error in Settings.toml: {}", err),
        Ok(config) => config,
    }
}

fn set_server(server_settings: Option<ServerConfig>) {
    let server_settings = server_settings.unwrap_or(ServerConfig {
        ip_addr: None,
        port: None,
        tls_cert: None,
        tls_key: None,
    });
    let port = server_settings.port.unwrap_or(3030);
    let ip = server_settings
        .ip_addr
        .unwrap_or_else(|| Ipv4Addr::new(127, 0, 0, 1));
    let socket_addr = SocketAddr::new(IpAddr::V4(ip), port);
    if server_settings.tls_key.is_some() && server_settings.tls_cert.is_some() {
        if cfg!(feature = "autoreload") {
            println!(
                "cargo:warning=TLS is enabled, but so is autoreload, autoreload cannot use tls"
            );
        } else {
            println!(
                "cargo:rustc-env=TLS_KEY={}",
                server_settings.tls_key.unwrap()
            );
            println!(
                "cargo:rustc-env=TLS_CERT={}",
                server_settings.tls_cert.unwrap()
            );
        }
    }
    println!("cargo:rustc-env=IP_ADDR={}", socket_addr);
}
//! Stdout based on the UART hooked up to FTDI or J-Link

use core::fmt;
use nb::block;
use riscv::interrupt;
use e310x_hal::{
    serial::{Serial, Tx, Rx},
    gpio::gpio0::{Pin17, Pin16},
    time::Bps,
    clock::Clocks,
    e310x::UART0,
    prelude::*
};


static mut STDOUT: Option<SerialWrapper> = None;


struct SerialWrapper(Tx<UART0>);

impl core::fmt::Write for SerialWrapper {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.as_bytes() {
            if *byte == '\n' as u8 {
                let res = block!(self.0.write('\r' as u8));

                if res.is_err() {
                    return Err(::core::fmt::Error);
                }
            }

            let res = block!(self.0.write(*byte));

            if res.is_err() {
                return Err(::core::fmt::Error);
            }
        }
        Ok(())
    }
}

/// Configures stdout
pub fn configure<X, Y>(
    uart: UART0, tx: Pin17<X>, rx: Pin16<Y>,
    baud_rate: Bps, clocks: Clocks
) -> Rx<UART0> {
    let tx = tx.into_iof0();
    let rx = rx.into_iof0();
    let serial = Serial::new(uart, (tx, rx), baud_rate, clocks);
    let (tx, rx) = serial.split();

    interrupt::free(|_| {
        unsafe {
            STDOUT.replace(SerialWrapper(tx));
        }
    });
    return rx;
}

/// Writes string to stdout
pub fn write_str(s: &str) {
    interrupt::free(|_| unsafe {
        if let Some(stdout) = STDOUT.as_mut() {
            let _ = stdout.write_str(s);
        }
    })
}

/// Writes formatted string to stdout
pub fn write_fmt(args: fmt::Arguments) {
    interrupt::free(|_| unsafe {
        if let Some(stdout) = STDOUT.as_mut() {
            let _ = stdout.write_fmt(args);
        }
    })
}

/// Macro for printing to the serial standard output
#[macro_export]
macro_rules! sprint {
    ($s:expr) => {
        $crate::stdout::write_str($s)
    };
    ($($tt:tt)*) => {
        $crate::stdout::write_fmt(format_args!($($tt)*))
    };
}

/// Macro for printing to the serial standard output, with a newline.
#[macro_export]
macro_rules! sprintln {
    () => {
        $crate::stdout::write_str("\n")
    };
    ($s:expr) => {
        $crate::stdout::write_str(concat!($s, "\n"))
    };
    ($s:expr, $($tt:tt)*) => {
        $crate::stdout::write_fmt(format_args!(concat!($s, "\n"), $($tt)*))
    };
}
use postgres::{Client, NoTls, Error};

pub struct Connector{
    host:String,
    port:u16,
    database:String,
    user:String,
    password:String,
    url:String,
}

impl Connector {
    pub fn new(user:String, host:String, database:String) -> Self{
        let url = format!("postgresql://{}@{}/{}",user,host,database);
        Connector{
            host,
            port:5432,
            database,
            user,
            password:String::from(""),
            url
        }
    }

    pub fn get_url(&self) -> &String{
        return &self.url;
    }

    pub fn client(&self) -> Result<Client,Error>{
        Ok(Client::connect(self.url.as_str(),NoTls)?)
    }
}
// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate sccache;

fn main() {
    sccache::main();
}
// Copyright 2017 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::mock_command::{CommandChild, RunCommand};
use blake3::Hasher as blake3_Hasher;
use byteorder::{BigEndian, ByteOrder};
use futures::{future, Future};
use futures_03::executor::ThreadPool;
use futures_03::future::TryFutureExt;
use futures_03::task;
use serde::Serialize;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::hash::Hasher;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{self, Stdio};
use std::time;
use std::time::Duration;

use crate::errors::*;

pub trait SpawnExt: task::SpawnExt {
    fn spawn_fn<F, T>(&self, f: F) -> SFuture<T>
    where
        F: FnOnce() -> Result<T> + std::marker::Send + 'static,
        T: std::marker::Send + 'static,
    {
        self.spawn_with_handle(async move { f() })
            .map(|f| Box::new(f.compat()) as _)
            .unwrap_or_else(f_err)
    }
}

impl<S: task::SpawnExt + ?Sized> SpawnExt for S {}

#[derive(Clone)]
pub struct Digest {
    inner: blake3_Hasher,
}

impl Digest {
    pub fn new() -> Digest {
        Digest {
            inner: blake3_Hasher::new(),
        }
    }

    /// Calculate the BLAKE3 digest of the contents of `path`, running
    /// the actual hash computation on a background thread in `pool`.
    pub fn file<T>(path: T, pool: &ThreadPool) -> SFuture<String>
    where
        T: AsRef<Path>,
    {
        Self::reader(path.as_ref().to_owned(), pool)
    }

    /// Calculate the BLAKE3 digest of the contents read from `reader`.
    pub fn reader_sync<R: Read>(mut reader: R) -> Result<String> {
        let mut m = Digest::new();
        // A buffer of 128KB should give us the best performance.
        // See https://eklitzke.org/efficient-file-copying-on-linux.
        let mut buffer = [0; 128 * 1024];
        loop {
            let count = reader.read(&mut buffer[..])?;
            if count == 0 {
                break;
            }
            m.update(&buffer[..count]);
        }
        Ok(m.finish())
    }

    /// Calculate the BLAKE3 digest of the contents of `path`, running
    /// the actual hash computation on a background thread in `pool`.
    pub fn reader(path: PathBuf, pool: &ThreadPool) -> SFuture<String> {
        Box::new(pool.spawn_fn(move || -> Result<_> {
            let reader = File::open(&path)
                .with_context(|| format!("Failed to open file for hashing: {:?}", path))?;
            Digest::reader_sync(reader)
        }))
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
    }

    pub fn finish(self) -> String {
        hex(self.inner.finalize().as_bytes())
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        s.push(hex(byte & 0xf));
        s.push(hex((byte >> 4) & 0xf));
    }
    return s;

    fn hex(byte: u8) -> char {
        match byte {
            0..=9 => (b'0' + byte) as char,
            _ => (b'a' + byte - 10) as char,
        }
    }
}

/// Calculate the digest of each file in `files` on background threads in
/// `pool`.
pub fn hash_all(files: &[PathBuf], pool: &ThreadPool) -> SFuture<Vec<String>> {
    let start = time::Instant::now();
    let count = files.len();
    let pool = pool.clone();
    Box::new(
        future::join_all(
            files
                .iter()
                .map(move |f| Digest::file(f, &pool))
                .collect::<Vec<_>>(),
        )
        .map(move |hashes| {
            trace!(
                "Hashed {} files in {}",
                count,
                fmt_duration_as_secs(&start.elapsed())
            );
            hashes
        }),
    )
}

/// Format `duration` as seconds with a fractional component.
pub fn fmt_duration_as_secs(duration: &Duration) -> String {
    format!("{}.{:03} s", duration.as_secs(), duration.subsec_millis())
}

/// If `input`, write it to `child`'s stdin while also reading `child`'s stdout and stderr, then wait on `child` and return its status and output.
///
/// This was lifted from `std::process::Child::wait_with_output` and modified
/// to also write to stdin.
fn wait_with_input_output<T>(mut child: T, input: Option<Vec<u8>>) -> SFuture<process::Output>
where
    T: CommandChild + 'static,
{
    use tokio_io::io::{read_to_end, write_all};
    let stdin = input.and_then(|i| {
        child
            .take_stdin()
            .map(|stdin| write_all(stdin, i).fcontext("failed to write stdin"))
    });
    let stdout = child
        .take_stdout()
        .map(|io| read_to_end(io, Vec::new()).fcontext("failed to read stdout"));
    let stderr = child
        .take_stderr()
        .map(|io| read_to_end(io, Vec::new()).fcontext("failed to read stderr"));

    // Finish writing stdin before waiting, because waiting drops stdin.
    let status = Future::and_then(stdin, |io| {
        drop(io);
        child.wait().fcontext("failed to wait for child")
    });

    Box::new(status.join3(stdout, stderr).map(|(status, out, err)| {
        let stdout = out.map(|p| p.1);
        let stderr = err.map(|p| p.1);
        process::Output {
            status,
            stdout: stdout.unwrap_or_default(),
            stderr: stderr.unwrap_or_default(),
        }
    }))
}

/// Run `command`, writing `input` to its stdin if it is `Some` and return the exit status and output.
///
/// If the command returns a non-successful exit status, an error of `SccacheError::ProcessError`
/// will be returned containing the process output.
pub fn run_input_output<C>(
    mut command: C,
    input: Option<Vec<u8>>,
) -> impl Future<Item = process::Output, Error = Error>
where
    C: RunCommand,
{
    let child = command
        .no_console()
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    child.and_then(|child| {
        wait_with_input_output(child, input).and_then(|output| {
            if output.status.success() {
                f_ok(output)
            } else {
                f_err(ProcessError(output))
            }
        })
    })
}

/// Write `data` to `writer` with bincode serialization, prefixed by a `u32` length.
pub fn write_length_prefixed_bincode<W, S>(mut writer: W, data: S) -> Result<()>
where
    W: Write,
    S: Serialize,
{
    let bytes = bincode::serialize(&data)?;
    let mut len = [0; 4];
    BigEndian::write_u32(&mut len, bytes.len() as u32);
    writer.write_all(&len)?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

pub trait OsStrExt {
    fn starts_with(&self, s: &str) -> bool;
    fn split_prefix(&self, s: &str) -> Option<OsString>;
}

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt as _OsStrExt;

#[cfg(unix)]
impl OsStrExt for OsStr {
    fn starts_with(&self, s: &str) -> bool {
        self.as_bytes().starts_with(s.as_bytes())
    }

    fn split_prefix(&self, s: &str) -> Option<OsString> {
        let bytes = self.as_bytes();
        if bytes.starts_with(s.as_bytes()) {
            Some(OsStr::from_bytes(&bytes[s.len()..]).to_owned())
        } else {
            None
        }
    }
}

#[cfg(windows)]
use std::os::windows::ffi::{OsStrExt as _OsStrExt, OsStringExt};

#[cfg(windows)]
impl OsStrExt for OsStr {
    fn starts_with(&self, s: &str) -> bool {
        // Attempt to interpret this OsStr as utf-16. This is a pretty "poor
        // man's" implementation, however, as it only handles a subset of
        // unicode characters in `s`. Currently that's sufficient, though, as
        // we're only calling `starts_with` with ascii string literals.
        let u16s = self.encode_wide();
        let mut utf8 = s.chars();

        for codepoint in u16s {
            let to_match = match utf8.next() {
                Some(ch) => ch,
                None => return true,
            };

            let to_match = to_match as u32;
            let codepoint = codepoint as u32;

            // UTF-16 encodes codepoints < 0xd7ff as just the raw value as a
            // u16, and that's all we're matching against. If the codepoint in
            // `s` is *over* this value then just assume it's not in `self`.
            //
            // If `to_match` is the same as the `codepoint` coming out of our
            // u16 iterator we keep going, otherwise we've found a mismatch.
            if to_match < 0xd7ff {
                if to_match != codepoint {
                    return false;
                }
            } else {
                return false;
            }
        }

        // If we ran out of characters to match, then the strings should be
        // equal, otherwise we've got more data to match in `s` so we didn't
        // start with `s`
        utf8.next().is_none()
    }

    fn split_prefix(&self, s: &str) -> Option<OsString> {
        // See comments in the above implementation for what's going on here
        let mut u16s = self.encode_wide().peekable();
        let mut utf8 = s.chars();

        while let Some(&codepoint) = u16s.peek() {
            let to_match = match utf8.next() {
                Some(ch) => ch,
                None => {
                    let codepoints = u16s.collect::<Vec<_>>();
                    return Some(OsString::from_wide(&codepoints));
                }
            };

            let to_match = to_match as u32;
            let codepoint = codepoint as u32;

            if to_match < 0xd7ff {
                if to_match != codepoint {
                    return None;
                }
            } else {
                return None;
            }
            u16s.next();
        }

        if utf8.next().is_none() {
            Some(OsString::new())
        } else {
            None
        }
    }
}

pub struct HashToDigest<'a> {
    pub digest: &'a mut Digest,
}

impl<'a> Hasher for HashToDigest<'a> {
    fn write(&mut self, bytes: &[u8]) {
        self.digest.update(bytes)
    }

    fn finish(&self) -> u64 {
        panic!("not supposed to be called");
    }
}

/// Turns a slice of environment var tuples into the type expected by Command::envs.
pub fn ref_env(env: &[(OsString, OsString)]) -> impl Iterator<Item = (&OsString, &OsString)> {
    env.iter().map(|&(ref k, ref v)| (k, v))
}

#[cfg(feature = "hyperx")]
pub use self::http_extension::{HeadersExt, RequestExt};

#[cfg(feature = "hyperx")]
mod http_extension {
    use http::header::HeaderValue;
    use std::fmt;

    pub trait HeadersExt {
        fn set<H>(&mut self, header: H)
        where
            H: hyperx::header::Header + fmt::Display;

        fn get_hyperx<H>(&self) -> Option<H>
        where
            H: hyperx::header::Header;
    }

    impl HeadersExt for http::HeaderMap {
        fn set<H>(&mut self, header: H)
        where
            H: hyperx::header::Header + fmt::Display,
        {
            self.insert(
                H::header_name(),
                HeaderValue::from_shared(header.to_string().into()).unwrap(),
            );
        }

        fn get_hyperx<H>(&self) -> Option<H>
        where
            H: hyperx::header::Header,
        {
            http::HeaderMap::get(self, H::header_name())
                .and_then(|header| H::parse_header(&header.as_bytes().into()).ok())
        }
    }

    pub trait RequestExt {
        fn set_header<H>(self, header: H) -> Self
        where
            H: hyperx::header::Header + fmt::Display;
    }

    impl RequestExt for http::request::Builder {
        fn set_header<H>(mut self, header: H) -> Self
        where
            H: hyperx::header::Header + fmt::Display,
        {
            self.header(
                H::header_name(),
                HeaderValue::from_shared(header.to_string().into()).unwrap(),
            );
            self
        }
    }

    impl RequestExt for http::response::Builder {
        fn set_header<H>(mut self, header: H) -> Self
        where
            H: hyperx::header::Header + fmt::Display,
        {
            self.header(
                H::header_name(),
                HeaderValue::from_shared(header.to_string().into()).unwrap(),
            );
            self
        }
    }

    #[cfg(feature = "reqwest")]
    impl RequestExt for ::reqwest::r#async::RequestBuilder {
        fn set_header<H>(self, header: H) -> Self
        where
            H: hyperx::header::Header + fmt::Display,
        {
            self.header(
                H::header_name(),
                HeaderValue::from_shared(header.to_string().into()).unwrap(),
            )
        }
    }

    #[cfg(feature = "reqwest")]
    impl RequestExt for ::reqwest::RequestBuilder {
        fn set_header<H>(self, header: H) -> Self
        where
            H: hyperx::header::Header + fmt::Display,
        {
            self.header(
                H::header_name(),
                HeaderValue::from_shared(header.to_string().into()).unwrap(),
            )
        }
    }
}

/// Pipe `cmd`'s stdio to `/dev/null`, unless a specific env var is set.
#[cfg(not(windows))]
pub fn daemonize() -> Result<()> {
    use daemonize::Daemonize;
    use std::env;
    use std::mem;

    match env::var("SCCACHE_NO_DAEMON") {
        Ok(ref val) if val == "1" => {}
        _ => {
            Daemonize::new().start().context("failed to daemonize")?;
        }
    }

    static mut PREV_SIGSEGV: *mut libc::sigaction = 0 as *mut _;
    static mut PREV_SIGBUS: *mut libc::sigaction = 0 as *mut _;
    static mut PREV_SIGILL: *mut libc::sigaction = 0 as *mut _;

    // We don't have a parent process any more once we've reached this point,
    // which means that no one's probably listening for our exit status.
    // In order to assist with debugging crashes of the server we configure our
    // rlimit to allow runtime dumps and we also install a signal handler for
    // segfaults which at least prints out what just happened.
    unsafe {
        match env::var("SCCACHE_ALLOW_CORE_DUMPS") {
            Ok(ref val) if val == "1" => {
                let rlim = libc::rlimit {
                    rlim_cur: libc::RLIM_INFINITY,
                    rlim_max: libc::RLIM_INFINITY,
                };
                libc::setrlimit(libc::RLIMIT_CORE, &rlim);
            }
            _ => {}
        }

        PREV_SIGSEGV = Box::into_raw(Box::new(mem::zeroed::<libc::sigaction>()));
        PREV_SIGBUS = Box::into_raw(Box::new(mem::zeroed::<libc::sigaction>()));
        PREV_SIGILL = Box::into_raw(Box::new(mem::zeroed::<libc::sigaction>()));
        let mut new: libc::sigaction = mem::zeroed();
        new.sa_sigaction = handler as usize;
        new.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        libc::sigaction(libc::SIGSEGV, &new, &mut *PREV_SIGSEGV);
        libc::sigaction(libc::SIGBUS, &new, &mut *PREV_SIGBUS);
        libc::sigaction(libc::SIGILL, &new, &mut *PREV_SIGILL);
    }

    return Ok(());

    extern "C" fn handler(
        signum: libc::c_int,
        _info: *mut libc::siginfo_t,
        _ptr: *mut libc::c_void,
    ) {
        use std::fmt::{Result, Write};

        struct Stderr;

        impl Write for Stderr {
            fn write_str(&mut self, s: &str) -> Result {
                unsafe {
                    let bytes = s.as_bytes();
                    libc::write(libc::STDERR_FILENO, bytes.as_ptr() as *const _, bytes.len());
                    Ok(())
                }
            }
        }

        unsafe {
            let _ = writeln!(Stderr, "signal {} received", signum);

            // Configure the old handler and then resume the program. This'll
            // likely go on to create a runtime dump if one's configured to be
            // created.
            match signum {
                libc::SIGBUS => libc::sigaction(signum, &*PREV_SIGBUS, std::ptr::null_mut()),
                libc::SIGILL => libc::sigaction(signum, &*PREV_SIGILL, std::ptr::null_mut()),
                _ => libc::sigaction(signum, &*PREV_SIGSEGV, std::ptr::null_mut()),
            };
        }
    }
}

/// This is a no-op on Windows.
#[cfg(windows)]
pub fn daemonize() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::OsStrExt;
    use std::ffi::{OsStr, OsString};

    #[test]
    fn simple_starts_with() {
        let a: &OsStr = "foo".as_ref();
        assert!(a.starts_with(""));
        assert!(a.starts_with("f"));
        assert!(a.starts_with("fo"));
        assert!(a.starts_with("foo"));
        assert!(!a.starts_with("foo2"));
        assert!(!a.starts_with("b"));
        assert!(!a.starts_with("b"));

        let a: &OsStr = "".as_ref();
        assert!(!a.starts_with("a"))
    }

    #[test]
    fn simple_strip_prefix() {
        let a: &OsStr = "foo".as_ref();

        assert_eq!(a.split_prefix(""), Some(OsString::from("foo")));
        assert_eq!(a.split_prefix("f"), Some(OsString::from("oo")));
        assert_eq!(a.split_prefix("fo"), Some(OsString::from("o")));
        assert_eq!(a.split_prefix("foo"), Some(OsString::from("")));
        assert_eq!(a.split_prefix("foo2"), None);
        assert_eq!(a.split_prefix("b"), None);
    }
}
use postgres::Client;
use crate::models::Model;

pub trait Connection{
    fn register_model(&self, model:impl Model);
}

pub struct PgConnection{
    client:Client,
}

impl PgConnection {
    pub fn new(client: Client) -> Self { Self { client } }
}


impl Connection for PgConnection {
    fn register_model(&self, model:impl Model) {
        model.test();
    }
}
use crate::inode::INode;
use fuse::FileType;
use std::{
    error::Error,
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
pub struct INodeTreeError {
    msg: String,
}

impl INodeTreeError {
    pub fn new(msg: &str) -> Self {
        Self {
            msg: msg.to_string(),
        }
    }
}

impl Error for INodeTreeError {}

impl fmt::Display for INodeTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "File error {}!", self.msg)
    }
}

pub struct INodeTree {
    pub nodes: Vec<INode>,
}

impl INodeTree {
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }

    pub fn next_ino(&self) -> u64 {
        match self.nodes.last() {
            Some(entry) => entry.ino + 1,
            _ => 1,
        }
    }

    pub fn remove(&mut self, ino: u64) {
        if let Some(pos) = self.nodes.iter().position(|e| e.ino == ino) {
            self.nodes.remove(pos);
        }
    }

    pub fn add(&mut self, inode: INode) {
        self.nodes.push(inode);
    }

    pub fn add_empty(&mut self, key: String, hash: blake3::Hash, parent: u64) -> u64 {
        let ino = self.next_ino();
        self.nodes.push(INode::new(
            &key,
            ino,
            0,
            UNIX_EPOCH,
            FileType::RegularFile,
            parent,
            hash,
        ));
        ino
    }

    pub fn add_from_keys(
        &mut self,
        nodes: Vec<(blake3::Hash, String, FileType)>,
        parent: Option<u64>,
    ) {
        let mut parent = parent.unwrap_or(1);
        for mut node in nodes {
            if node.2 == FileType::Directory {
                self.add_inode_dir(&node.1, None);
                continue;
            }
            if node.1.contains("/") {
                let pos_of_last = node.1.rfind("/").unwrap();
                let folders = &node.1[..pos_of_last];
                parent = self.add_inode_dir(folders, None);
                node.1 = node.1[pos_of_last + 1..].to_string();
            }
            self.add_empty(node.1, node.0, parent);
        }
    }

    pub fn add_inode_dir(&mut self, dir: &str, parent: Option<u64>) -> u64 {
        let folders = dir.split("/");
        let mut parent = parent.unwrap_or(1);
        for folder in folders {
            match self.get_inode_from_key_parent(folder, parent) {
                Ok(entry) => {
                    parent = entry.ino;
                    continue;
                }
                _ => {}
            };
            let ino = self.next_ino();

            let next_ino = ino;
            self.add(INode::new(
                folder,
                ino,
                0,
                SystemTime::now(),
                FileType::Directory,
                parent,
                blake3::hash(b""),
            ));
            parent = next_ino;
        }
        parent
    }

    pub fn get_children(&self, ino: u64) -> Vec<&INode> {
        self.nodes.iter().filter(|e| e.parent == ino).collect()
    }
    fn get_subtree(&self, ino: u64) -> Vec<&INode> {
        let mut entries = self.get_children(ino);
        for entry in entries.clone() {
            let mut children = self.get_subtree(entry.ino);
            entries.append(&mut children);
        }
        entries
    }

    pub fn get_full_path(&self, ino: u64) -> Result<String, INodeTreeError> {
        let node = self.get_inode_from_ino(ino)?;
        if node.ino == 1 {
            Ok("".to_string())
        } else {
            let parent_path = self.get_full_path(node.parent)?;
            if node.parent != 1 {
                Ok(format!("{}/{}", parent_path, node.key).to_owned())
            } else {
                Ok(node.key.clone())
            }
        }
    }

    pub fn get_inode_from_ino(&self, ino: u64) -> Result<&INode, INodeTreeError> {
        let file: Vec<&INode> = self.nodes.iter().filter(|e| e.ino == ino).collect();
        match file.len() {
            0 => Err(INodeTreeError::new("File Not Found")),
            1 => Ok(file.first().unwrap()),
            _ => panic!("Error: Duplicate keys"),
        }
    }

    pub fn get_inode_from_key_parent(
        &self,
        name: &str,
        parent_ino: u64,
    ) -> Result<&INode, INodeTreeError> {
        let file: Vec<&INode> = self
            .nodes
            .iter()
            .filter(|e| e.parent == parent_ino)
            .filter(|e| e.key == name)
            .collect();
        match file.len() {
            0 => Err(INodeTreeError::new("File Not Found")),
            1 => Ok(file.first().unwrap()),
            _ => panic!("Error: Duplicate keys"),
        }
    }

    pub fn update_inode_hash(
        &mut self,
        ino: u64,
        hash: blake3::Hash,
    ) -> Result<(), INodeTreeError> {
        if let Some(mut inode) = self
            .nodes
            .iter_mut()
            .filter(|e| e.ino == ino)
            .collect::<Vec<&mut INode>>()
            .first_mut()
        {
            inode.hash = hash;
            Ok(())
        } else {
            Err(INodeTreeError::new("File Not Found"))
        }
    }

    pub fn update_inode_attr(
        &mut self,
        ino: u64,
        mtime: SystemTime,
        size: u64,
    ) -> Result<(), INodeTreeError> {
        if let Some(mut inode) = self
            .nodes
            .iter_mut()
            .filter(|e| e.ino == ino)
            .collect::<Vec<&mut INode>>()
            .first_mut()
        {
            inode.mtime = mtime;
            inode.size = size;
            Ok(())
        } else {
            Err(INodeTreeError::new("File Not Found"))
        }
    }

    pub fn write_all_to_string(&self) -> String {
        let mut final_string = String::new();
        for node in &self.nodes {
            let kind = match node.kind {
                FileType::Directory => "dir".to_string(),
                FileType::RegularFile => "file".to_string(),
                _ => {
                    continue;
                }
            };
            let key = format!("\"{}\"", self.get_full_path(node.ino).unwrap());
            let hash = node.hash.to_hex();
            let string = format!("{}\t{}\t{}\n", kind, hash, key);
            final_string = final_string + &string;
        }
        final_string
    }

    pub fn get_hash_list(&self) -> Vec<blake3::Hash> {
        let mut hashes = vec![];
        for node in &self.nodes {
            if node.kind == FileType::Directory {
                continue;
            }
            hashes.push(node.hash.clone());
        }
        hashes
    }

    pub fn get_root_parent(&self, ino:u64) -> Result<&INode,INodeTreeError>{
        let node = self.get_inode_from_ino(ino)?;
        match node.parent{
            1 => Ok(node),
            _ => Ok(self.get_root_parent(node.parent)?)
        }
    }
}
pub mod keys;
pub mod keys_grpc;
//! HTTP specific body utilities.
//!
//! This module contains traits and helper types to work with http bodies. Most
//! of the types in this module are based around [`http_body::Body`].

use crate::{Error, Status};
use bytes::{Buf, Bytes};
use http_body::Body as HttpBody;
use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

/// A trait alias for [`http_body::Body`].
pub trait Body: sealed::Sealed + Send + Sync {
    /// The body data type.
    type Data: Buf;
    /// The errors produced from the body.
    type Error: Into<Error>;

    /// Check if the stream is over or not.
    ///
    /// Reference [`http_body::Body::is_end_stream`].
    fn is_end_stream(&self) -> bool;

    /// Poll for more data from the body.
    ///
    /// Reference [`http_body::Body::poll_data`].
    fn poll_data(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>>;

    /// Poll for the trailing headers.
    ///
    /// Reference [`http_body::Body::poll_trailers`].
    fn poll_trailers(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<http::HeaderMap>, Self::Error>>;
}

impl<T> Body for T
where
    T: HttpBody + Send + Sync + 'static,
    T::Error: Into<Error>,
{
    type Data = T::Data;
    type Error = T::Error;

    fn is_end_stream(&self) -> bool {
        HttpBody::is_end_stream(self)
    }

    fn poll_data(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>> {
        HttpBody::poll_data(self, cx)
    }

    fn poll_trailers(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<http::HeaderMap>, Self::Error>> {
        HttpBody::poll_trailers(self, cx)
    }
}

impl<T> sealed::Sealed for T
where
    T: HttpBody,
    T::Error: Into<Error>,
{
}

mod sealed {
    pub trait Sealed {}
}

/// A type erased http body.
pub struct BoxBody {
    inner: Pin<Box<dyn Body<Data = Bytes, Error = Status> + Send + Sync + 'static>>,
}

struct MapBody<B>(B);

impl BoxBody {
    /// Create a new `BoxBody` mapping item and error to the default types.
    pub fn new<B>(inner: B) -> Self
    where
        B: Body<Data = Bytes, Error = Status> + Send + Sync + 'static,
    {
        BoxBody {
            inner: Box::pin(inner),
        }
    }

    /// Create a new `BoxBody` mapping item and error to the default types.
    pub fn map_from<B>(inner: B) -> Self
    where
        B: Body + Send + Sync + 'static,
        B::Error: Into<crate::Error>,
    {
        BoxBody {
            inner: Box::pin(MapBody(inner)),
        }
    }

    /// Create a new `BoxBody` that is empty.
    pub fn empty() -> Self {
        BoxBody {
            inner: Box::pin(EmptyBody::default()),
        }
    }
}

impl HttpBody for BoxBody {
    type Data = Bytes;
    type Error = Status;

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn poll_data(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>> {
        Body::poll_data(self.inner.as_mut(), cx)
    }

    fn poll_trailers(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<http::HeaderMap>, Self::Error>> {
        Body::poll_trailers(self.inner.as_mut(), cx)
    }
}

impl<B> HttpBody for MapBody<B>
where
    B: Body,
    B::Error: Into<crate::Error>,
{
    type Data = Bytes;
    type Error = Status;

    fn is_end_stream(&self) -> bool {
        self.0.is_end_stream()
    }

    fn poll_data(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>> {
        let v = unsafe {
            let me = self.get_unchecked_mut();
            Pin::new_unchecked(&mut me.0).poll_data(cx)
        };
        match futures_util::ready!(v) {
            Some(Ok(mut i)) => Poll::Ready(Some(Ok(i.to_bytes()))),
            Some(Err(e)) => {
                let err = Status::map_error(e.into());
                Poll::Ready(Some(Err(err)))
            }
            None => Poll::Ready(None),
        }
    }

    fn poll_trailers(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<http::HeaderMap>, Self::Error>> {
        let v = unsafe {
            let me = self.get_unchecked_mut();
            Pin::new_unchecked(&mut me.0).poll_trailers(cx)
        };

        let v = futures_util::ready!(v).map_err(|e| Status::from_error(&*e.into()));
        Poll::Ready(v)
    }
}

impl fmt::Debug for BoxBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxBody").finish()
    }
}

#[derive(Debug, Default)]
struct EmptyBody {
    _p: (),
}

impl HttpBody for EmptyBody {
    type Data = Bytes;
    type Error = Status;

    fn is_end_stream(&self) -> bool {
        true
    }

    fn poll_data(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>> {
        Poll::Ready(None)
    }

    fn poll_trailers(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Option<http::HeaderMap>, Self::Error>> {
        Poll::Ready(Ok(None))
    }
}
// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::errors::*;
use crate::protocol::{Request, Response};
use crate::util;
use byteorder::{BigEndian, ByteOrder};
use retry::{delay::Fixed, retry};
use std::io::{self, BufReader, BufWriter, Read};
use std::net::TcpStream;

/// A connection to an sccache server.
pub struct ServerConnection {
    /// A reader for the socket connected to the server.
    reader: BufReader<TcpStream>,
    /// A writer for the socket connected to the server.
    writer: BufWriter<TcpStream>,
}

impl ServerConnection {
    /// Create a new connection using `stream`.
    pub fn new(stream: TcpStream) -> io::Result<ServerConnection> {
        let writer = stream.try_clone()?;
        Ok(ServerConnection {
            reader: BufReader::new(stream),
            writer: BufWriter::new(writer),
        })
    }

    /// Send `request` to the server, read and return a `Response`.
    pub fn request(&mut self, request: Request) -> Result<Response> {
        trace!("ServerConnection::request");
        util::write_length_prefixed_bincode(&mut self.writer, request)?;
        trace!("ServerConnection::request: sent request");
        self.read_one_response()
    }

    /// Read a single `Response` from the server.
    pub fn read_one_response(&mut self) -> Result<Response> {
        trace!("ServerConnection::read_one_response");
        let mut bytes = [0; 4];
        self.reader
            .read_exact(&mut bytes)
            .context("Failed to read response header")?;
        let len = BigEndian::read_u32(&bytes);
        trace!("Should read {} more bytes", len);
        let mut data = vec![0; len as usize];
        self.reader.read_exact(&mut data)?;
        trace!("Done reading");
        Ok(bincode::deserialize(&data)?)
    }
}

/// Establish a TCP connection to an sccache server listening on `port`.
pub fn connect_to_server(port: u16) -> io::Result<ServerConnection> {
    trace!("connect_to_server({})", port);
    let stream = TcpStream::connect(("127.0.0.1", port))?;
    ServerConnection::new(stream)
}

/// Attempt to establish a TCP connection to an sccache server listening on `port`.
///
/// If the connection fails, retry a few times.
pub fn connect_with_retry(port: u16) -> io::Result<ServerConnection> {
    trace!("connect_with_retry({})", port);
    // TODOs:
    // * Pass the server Child in here, so we can stop retrying
    //   if the process exited.
    // * Send a pipe handle to the server process so it can notify
    //   us once it starts the server instead of us polling.
    match retry(Fixed::from_millis(500).take(10), || connect_to_server(port)) {
        Ok(conn) => Ok(conn),
        _ => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Connection to server timed out",
        )),
    }
}
// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub use anyhow::{anyhow, bail, Context, Error};
use futures::future;
use futures::Future;
use std::boxed::Box;
use std::fmt::Display;
use std::process;

// We use `anyhow` for error handling.
// - Use `context()`/`with_context()` to annotate errors.
// - Use `anyhow!` with a string to create a new `anyhow::Error`.
// - The error types below (`BadHttpStatusError`, etc.) are internal ones that
//   need to be checked at points other than the outermost error-checking
//   layer.
// - There are some combinators below for working with futures.

#[cfg(feature = "hyper")]
#[derive(Debug)]
pub struct BadHttpStatusError(pub hyper::StatusCode);

#[derive(Debug)]
pub struct HttpClientError(pub String);

#[derive(Debug)]
pub struct ProcessError(pub process::Output);

#[cfg(feature = "hyper")]
impl std::error::Error for BadHttpStatusError {}

impl std::error::Error for HttpClientError {}

impl std::error::Error for ProcessError {}

#[cfg(feature = "hyper")]
impl std::fmt::Display for BadHttpStatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "didn't get a successful HTTP status, got `{}`", self.0)
    }
}

impl std::fmt::Display for HttpClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "didn't get a successful HTTP status, got `{}`", self.0)
    }
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.0.stderr))
    }
}

pub type Result<T> = anyhow::Result<T>;

pub type SFuture<T> = Box<dyn Future<Item = T, Error = Error>>;
pub type SFutureSend<T> = Box<dyn Future<Item = T, Error = Error> + Send>;

pub trait FutureContext<T> {
    fn fcontext<C>(self, context: C) -> SFuture<T>
    where
        C: Display + Send + Sync + 'static;

    fn fwith_context<C, CB>(self, callback: CB) -> SFuture<T>
    where
        CB: FnOnce() -> C + 'static,
        C: Display + Send + Sync + 'static;
}

impl<F> FutureContext<F::Item> for F
where
    F: Future + 'static,
    F::Error: Into<Error> + Send + Sync,
{
    fn fcontext<C>(self, context: C) -> SFuture<F::Item>
    where
        C: Display + Send + Sync + 'static,
    {
        Box::new(self.then(|r| r.map_err(F::Error::into).context(context)))
    }

    fn fwith_context<C, CB>(self, callback: CB) -> SFuture<F::Item>
    where
        CB: FnOnce() -> C + 'static,
        C: Display + Send + Sync + 'static,
    {
        Box::new(self.then(|r| r.map_err(F::Error::into).context(callback())))
    }
}

/// Like `try`, but returns an SFuture instead of a Result.
macro_rules! ftry {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Box::new($crate::futures::future::err(e.into())) as SFuture<_>,
        }
    };
}

#[cfg(feature = "dist-client")]
macro_rules! ftry_send {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Box::new($crate::futures::future::err(e)) as SFutureSend<_>,
        }
    };
}

pub fn f_ok<T>(t: T) -> SFuture<T>
where
    T: 'static,
{
    Box::new(future::ok(t))
}

pub fn f_err<T, E>(e: E) -> SFuture<T>
where
    T: 'static,
    E: Into<Error>,
{
    Box::new(future::err(e.into()))
}
// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use directories::ProjectDirs;
use regex::Regex;
use serde::de::{Deserialize, DeserializeOwned, Deserializer};
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
use serde::ser::{Serialize, Serializer};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::str::FromStr;
use std::sync::Mutex;

use crate::errors::*;

lazy_static! {
    static ref CACHED_CONFIG_PATH: PathBuf = CachedConfig::file_config_path();
    static ref CACHED_CONFIG: Mutex<Option<CachedFileConfig>> = Mutex::new(None);
}

const ORGANIZATION: &str = "Mozilla";
const APP_NAME: &str = "sccache";
const DIST_APP_NAME: &str = "sccache-dist-client";
const TEN_GIGS: u64 = 10 * 1024 * 1024 * 1024;

const MOZILLA_OAUTH_PKCE_CLIENT_ID: &str = "F1VVD6nRTckSVrviMRaOdLBWIk1AvHYo";
// The sccache audience is an API set up in auth0 for sccache to allow 7 day expiry,
// the openid scope allows us to query the auth0 /userinfo endpoint which contains
// group information due to Mozilla rules.
const MOZILLA_OAUTH_PKCE_AUTH_URL: &str =
    "https://auth.mozilla.auth0.com/authorize?audience=sccache&scope=openid%20profile";
const MOZILLA_OAUTH_PKCE_TOKEN_URL: &str = "https://auth.mozilla.auth0.com/oauth/token";

pub const INSECURE_DIST_CLIENT_TOKEN: &str = "dangerously_insecure_client";

// Unfortunately this means that nothing else can use the sccache cache dir as
// this top level directory is used directly to store sccache cached objects...
pub fn default_disk_cache_dir() -> PathBuf {
    ProjectDirs::from("", ORGANIZATION, APP_NAME)
        .expect("Unable to retrieve disk cache directory")
        .cache_dir()
        .to_owned()
}
// ...whereas subdirectories are used of this one
pub fn default_dist_cache_dir() -> PathBuf {
    ProjectDirs::from("", ORGANIZATION, DIST_APP_NAME)
        .expect("Unable to retrieve dist cache directory")
        .cache_dir()
        .to_owned()
}

fn default_disk_cache_size() -> u64 {
    TEN_GIGS
}
fn default_toolchain_cache_size() -> u64 {
    TEN_GIGS
}

pub fn parse_size(val: &str) -> Option<u64> {
    let re = Regex::new(r"^(\d+)([KMGT])$").expect("Fixed regex parse failure");
    re.captures(val)
        .and_then(|caps| {
            caps.get(1)
                .and_then(|size| u64::from_str(size.as_str()).ok())
                .map(|size| (size, caps.get(2)))
        })
        .and_then(|(size, suffix)| match suffix.map(|s| s.as_str()) {
            Some("K") => Some(1024 * size),
            Some("M") => Some(1024 * 1024 * size),
            Some("G") => Some(1024 * 1024 * 1024 * size),
            Some("T") => Some(1024 * 1024 * 1024 * 1024 * size),
            _ => None,
        })
}

#[cfg(any(feature = "dist-client", feature = "dist-server"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HTTPUrl(reqwest::Url);
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
impl Serialize for HTTPUrl {
    fn serialize<S>(&self, serializer: S) -> StdResult<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.0.as_str())
    }
}
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
impl<'a> Deserialize<'a> for HTTPUrl {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        let helper: String = Deserialize::deserialize(deserializer)?;
        let url = parse_http_url(&helper).map_err(D::Error::custom)?;
        Ok(HTTPUrl(url))
    }
}
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
fn parse_http_url(url: &str) -> Result<reqwest::Url> {
    use std::net::SocketAddr;
    let url = if let Ok(sa) = url.parse::<SocketAddr>() {
        warn!("Url {} has no scheme, assuming http", url);
        reqwest::Url::parse(&format!("http://{}", sa))
    } else {
        reqwest::Url::parse(url)
    }?;
    if url.scheme() != "http" && url.scheme() != "https" {
        bail!("url not http or https")
    }
    // TODO: relative url handling just hasn't been implemented and tested
    if url.path() != "/" {
        bail!("url has a relative path (currently unsupported)")
    }
    Ok(url)
}
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
impl HTTPUrl {
    pub fn from_url(u: reqwest::Url) -> Self {
        HTTPUrl(u)
    }
    pub fn to_url(&self) -> reqwest::Url {
        self.0.clone()
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AzureCacheConfig;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct DiskCacheConfig {
    pub dir: PathBuf,
    // TODO: use deserialize_with to allow human-readable sizes in toml
    pub size: u64,
}

impl Default for DiskCacheConfig {
    fn default() -> Self {
        DiskCacheConfig {
            dir: default_disk_cache_dir(),
            size: default_disk_cache_size(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum GCSCacheRWMode {
    #[serde(rename = "READ_ONLY")]
    ReadOnly,
    #[serde(rename = "READ_WRITE")]
    ReadWrite,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GCSCacheConfig {
    pub bucket: String,
    pub cred_path: Option<PathBuf>,
    pub url: Option<String>,
    pub rw_mode: GCSCacheRWMode,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemcachedCacheConfig {
    pub url: String,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedisCacheConfig {
    pub url: String,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S3CacheConfig {
    pub bucket: String,
    pub endpoint: String,
    pub use_ssl: bool,
    pub key_prefix: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CacheType {
    Azure(AzureCacheConfig),
    GCS(GCSCacheConfig),
    Memcached(MemcachedCacheConfig),
    Redis(RedisCacheConfig),
    S3(S3CacheConfig),
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfigs {
    pub azure: Option<AzureCacheConfig>,
    pub disk: Option<DiskCacheConfig>,
    pub gcs: Option<GCSCacheConfig>,
    pub memcached: Option<MemcachedCacheConfig>,
    pub redis: Option<RedisCacheConfig>,
    pub s3: Option<S3CacheConfig>,
}

impl CacheConfigs {
    /// Return a vec of the available cache types in an arbitrary but
    /// consistent ordering
    fn into_vec_and_fallback(self) -> (Vec<CacheType>, DiskCacheConfig) {
        let CacheConfigs {
            azure,
            disk,
            gcs,
            memcached,
            redis,
            s3,
        } = self;

        let caches = s3
            .map(CacheType::S3)
            .into_iter()
            .chain(redis.map(CacheType::Redis))
            .chain(memcached.map(CacheType::Memcached))
            .chain(gcs.map(CacheType::GCS))
            .chain(azure.map(CacheType::Azure))
            .collect();
        let fallback = disk.unwrap_or_else(Default::default);

        (caches, fallback)
    }

    /// Override self with any existing fields from other
    fn merge(&mut self, other: Self) {
        let CacheConfigs {
            azure,
            disk,
            gcs,
            memcached,
            redis,
            s3,
        } = other;

        if azure.is_some() {
            self.azure = azure
        }
        if disk.is_some() {
            self.disk = disk
        }
        if gcs.is_some() {
            self.gcs = gcs
        }
        if memcached.is_some() {
            self.memcached = memcached
        }
        if redis.is_some() {
            self.redis = redis
        }
        if s3.is_some() {
            self.s3 = s3
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum DistToolchainConfig {
    #[serde(rename = "no_dist")]
    NoDist { compiler_executable: PathBuf },
    #[serde(rename = "path_override")]
    PathOverride {
        compiler_executable: PathBuf,
        archive: PathBuf,
        archive_compiler_executable: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type")]
pub enum DistAuth {
    #[serde(rename = "token")]
    Token { token: String },
    #[serde(rename = "oauth2_code_grant_pkce")]
    Oauth2CodeGrantPKCE {
        client_id: String,
        auth_url: String,
        token_url: String,
    },
    #[serde(rename = "oauth2_implicit")]
    Oauth2Implicit { client_id: String, auth_url: String },
}

// Convert a type = "mozilla" immediately into an actual oauth configuration
// https://github.com/serde-rs/serde/issues/595 could help if implemented
impl<'a> Deserialize<'a> for DistAuth {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        #[serde(tag = "type")]
        pub enum Helper {
            #[serde(rename = "token")]
            Token { token: String },
            #[serde(rename = "mozilla")]
            Mozilla,
            #[serde(rename = "oauth2_code_grant_pkce")]
            Oauth2CodeGrantPKCE {
                client_id: String,
                auth_url: String,
                token_url: String,
            },
            #[serde(rename = "oauth2_implicit")]
            Oauth2Implicit { client_id: String, auth_url: String },
        }

        let helper: Helper = Deserialize::deserialize(deserializer)?;

        Ok(match helper {
            Helper::Token { token } => DistAuth::Token { token },
            Helper::Mozilla => DistAuth::Oauth2CodeGrantPKCE {
                client_id: MOZILLA_OAUTH_PKCE_CLIENT_ID.to_owned(),
                auth_url: MOZILLA_OAUTH_PKCE_AUTH_URL.to_owned(),
                token_url: MOZILLA_OAUTH_PKCE_TOKEN_URL.to_owned(),
            },
            Helper::Oauth2CodeGrantPKCE {
                client_id,
                auth_url,
                token_url,
            } => DistAuth::Oauth2CodeGrantPKCE {
                client_id,
                auth_url,
                token_url,
            },
            Helper::Oauth2Implicit {
                client_id,
                auth_url,
            } => DistAuth::Oauth2Implicit {
                client_id,
                auth_url,
            },
        })
    }
}

impl Default for DistAuth {
    fn default() -> Self {
        DistAuth::Token {
            token: INSECURE_DIST_CLIENT_TOKEN.to_owned(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct DistConfig {
    pub auth: DistAuth,
    #[cfg(any(feature = "dist-client", feature = "dist-server"))]
    pub scheduler_url: Option<HTTPUrl>,
    #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
    pub scheduler_url: Option<String>,
    pub cache_dir: PathBuf,
    pub toolchains: Vec<DistToolchainConfig>,
    pub toolchain_cache_size: u64,
    pub rewrite_includes_only: bool,
}

impl Default for DistConfig {
    fn default() -> Self {
        Self {
            auth: Default::default(),
            scheduler_url: Default::default(),
            cache_dir: default_dist_cache_dir(),
            toolchains: Default::default(),
            toolchain_cache_size: default_toolchain_cache_size(),
            rewrite_includes_only: false,
        }
    }
}

// TODO: fields only pub for tests
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub cache: CacheConfigs,
    pub dist: DistConfig,
}

// If the file doesn't exist or we can't read it, log the issue and proceed. If the
// config exists but doesn't parse then something is wrong - return an error.
pub fn try_read_config_file<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    debug!("Attempting to read config file at {:?}", path);
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            debug!("Couldn't open config file: {}", e);
            return Ok(None);
        }
    };

    let mut string = String::new();
    match file.read_to_string(&mut string) {
        Ok(_) => (),
        Err(e) => {
            warn!("Failed to read config file: {}", e);
            return Ok(None);
        }
    }

    let res = if path.extension().map_or(false, |e| e == "json") {
        serde_json::from_str(&string)
            .with_context(|| format!("Failed to load json config file from {}", path.display()))?
    } else {
        toml::from_str(&string)
            .with_context(|| format!("Failed to load toml config file from {}", path.display()))?
    };

    Ok(Some(res))
}

#[derive(Debug)]
pub struct EnvConfig {
    cache: CacheConfigs,
}

fn config_from_env() -> EnvConfig {
    let s3 = env::var("SCCACHE_BUCKET").ok().map(|bucket| {
        let endpoint = match env::var("SCCACHE_ENDPOINT") {
            Ok(endpoint) => format!("{}/{}", endpoint, bucket),
            _ => match env::var("SCCACHE_REGION") {
                Ok(ref region) if region != "us-east-1" => {
                    format!("{}.s3-{}.amazonaws.com", bucket, region)
                }
                _ => format!("{}.s3.amazonaws.com", bucket),
            },
        };
        let use_ssl = env::var("SCCACHE_S3_USE_SSL")
            .ok()
            .filter(|value| value != "off")
            .is_some();
        let key_prefix = env::var("SCCACHE_S3_KEY_PREFIX")
            .ok()
            .as_ref()
            .map(|s| s.trim_end_matches('/'))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_owned() + "/")
            .unwrap_or_default();

        S3CacheConfig {
            bucket,
            endpoint,
            use_ssl,
            key_prefix,
        }
    });

    let redis = env::var("SCCACHE_REDIS")
        .ok()
        .map(|url| RedisCacheConfig { url });

    let memcached = env::var("SCCACHE_MEMCACHED")
        .ok()
        .map(|url| MemcachedCacheConfig { url });

    let gcs = env::var("SCCACHE_GCS_BUCKET").ok().map(|bucket| {
        let url = env::var("SCCACHE_GCS_CREDENTIALS_URL").ok();
        let cred_path = env::var_os("SCCACHE_GCS_KEY_PATH").map(PathBuf::from);

        if url.is_some() && cred_path.is_some() {
            warn!("Both SCCACHE_GCS_CREDENTIALS_URL and SCCACHE_GCS_KEY_PATH are set");
            warn!("You should set only one of them!");
            warn!("SCCACHE_GCS_KEY_PATH will take precedence");
        }

        let rw_mode = match env::var("SCCACHE_GCS_RW_MODE").as_ref().map(String::as_str) {
            Ok("READ_ONLY") => GCSCacheRWMode::ReadOnly,
            Ok("READ_WRITE") => GCSCacheRWMode::ReadWrite,
            // TODO: unsure if these should warn during the configuration loading
            // or at the time when they're actually used to connect to GCS
            Ok(_) => {
                warn!("Invalid SCCACHE_GCS_RW_MODE-- defaulting to READ_ONLY.");
                GCSCacheRWMode::ReadOnly
            }
            _ => {
                warn!("No SCCACHE_GCS_RW_MODE specified-- defaulting to READ_ONLY.");
                GCSCacheRWMode::ReadOnly
            }
        };
        GCSCacheConfig {
            bucket,
            cred_path,
            url,
            rw_mode,
        }
    });

    let azure = env::var("SCCACHE_AZURE_CONNECTION_STRING")
        .ok()
        .map(|_| AzureCacheConfig);

    let disk_dir = env::var_os("SCCACHE_DIR").map(PathBuf::from);
    let disk_sz = env::var("SCCACHE_CACHE_SIZE")
        .ok()
        .and_then(|v| parse_size(&v));

    let disk = if disk_dir.is_some() || disk_sz.is_some() {
        Some(DiskCacheConfig {
            dir: disk_dir.unwrap_or_else(default_disk_cache_dir),
            size: disk_sz.unwrap_or_else(default_disk_cache_size),
        })
    } else {
        None
    };

    let cache = CacheConfigs {
        azure,
        disk,
        gcs,
        memcached,
        redis,
        s3,
    };

    EnvConfig { cache }
}

// The directories crate changed the location of `config_dir` on macos in version 3,
// so we also check the config in `preference_dir` (new in that version), which
// corresponds to the old location, for compatibility with older setups.
fn config_file(env_var: &str, leaf: &str) -> PathBuf {
    if let Some(env_value) = env::var_os(env_var) {
        return env_value.into();
    }
    let dirs =
        ProjectDirs::from("", ORGANIZATION, APP_NAME).expect("Unable to get config directory");
    // If the new location exists, use that.
    let path = dirs.config_dir().join(leaf);
    if path.exists() {
        return path;
    }
    // If the old location exists, use that.
    let path = dirs.preference_dir().join(leaf);
    if path.exists() {
        return path;
    }
    // Otherwise, use the new location.
    dirs.config_dir().join(leaf)
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Config {
    pub caches: Vec<CacheType>,
    pub fallback_cache: DiskCacheConfig,
    pub dist: DistConfig,
}

impl Config {
    pub fn load() -> Result<Config> {
        let env_conf = config_from_env();

        let file_conf_path = config_file("SCCACHE_CONF", "config");
        let file_conf = try_read_config_file(&file_conf_path)
            .context("Failed to load config file")?
            .unwrap_or_default();

        Ok(Config::from_env_and_file_configs(env_conf, file_conf))
    }

    fn from_env_and_file_configs(env_conf: EnvConfig, file_conf: FileConfig) -> Config {
        let mut conf_caches: CacheConfigs = Default::default();

        let FileConfig { cache, dist } = file_conf;
        conf_caches.merge(cache);

        let EnvConfig { cache } = env_conf;
        conf_caches.merge(cache);

        let (caches, fallback_cache) = conf_caches.into_vec_and_fallback();
        Config {
            caches,
            fallback_cache,
            dist,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct CachedDistConfig {
    pub auth_tokens: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct CachedFileConfig {
    pub dist: CachedDistConfig,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct CachedConfig(());

impl CachedConfig {
    pub fn load() -> Result<Self> {
        let mut cached_file_config = CACHED_CONFIG.lock().unwrap();

        if cached_file_config.is_none() {
            let cfg = Self::load_file_config().context("Unable to initialise cached config")?;
            *cached_file_config = Some(cfg)
        }
        Ok(CachedConfig(()))
    }
    pub fn reload() -> Result<Self> {
        {
            let mut cached_file_config = CACHED_CONFIG.lock().unwrap();
            *cached_file_config = None;
        };
        Self::load()
    }
    pub fn with<F: FnOnce(&CachedFileConfig) -> T, T>(&self, f: F) -> T {
        let cached_file_config = CACHED_CONFIG.lock().unwrap();
        let cached_file_config = cached_file_config.as_ref().unwrap();

        f(&cached_file_config)
    }
    pub fn with_mut<F: FnOnce(&mut CachedFileConfig)>(&self, f: F) -> Result<()> {
        let mut cached_file_config = CACHED_CONFIG.lock().unwrap();
        let cached_file_config = cached_file_config.as_mut().unwrap();

        let mut new_config = cached_file_config.clone();
        f(&mut new_config);
        Self::save_file_config(&new_config)?;
        *cached_file_config = new_config;
        Ok(())
    }

    fn file_config_path() -> PathBuf {
        config_file("SCCACHE_CACHED_CONF", "cached-config")
    }
    fn load_file_config() -> Result<CachedFileConfig> {
        let file_conf_path = &*CACHED_CONFIG_PATH;

        if !file_conf_path.exists() {
            let file_conf_dir = file_conf_path
                .parent()
                .expect("Cached conf file has no parent directory");
            if !file_conf_dir.is_dir() {
                fs::create_dir_all(file_conf_dir)
                    .context("Failed to create dir to hold cached config")?
            }
            Self::save_file_config(&Default::default()).with_context(|| {
                format!(
                    "Unable to create cached config file at {}",
                    file_conf_path.display()
                )
            })?
        }
        try_read_config_file(&file_conf_path)
            .context("Failed to load cached config file")?
            .with_context(|| format!("Failed to load from {}", file_conf_path.display()))
    }
    fn save_file_config(c: &CachedFileConfig) -> Result<()> {
        let file_conf_path = &*CACHED_CONFIG_PATH;
        let mut file = File::create(file_conf_path).context("Could not open config for writing")?;
        file.write_all(&toml::to_vec(c).unwrap())
            .map_err(Into::into)
    }
}

#[cfg(feature = "dist-server")]
pub mod scheduler {
    use std::net::SocketAddr;
    use std::path::Path;

    use crate::errors::*;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    #[serde(deny_unknown_fields)]
    pub enum ClientAuth {
        #[serde(rename = "DANGEROUSLY_INSECURE")]
        Insecure,
        #[serde(rename = "token")]
        Token { token: String },
        #[serde(rename = "jwt_validate")]
        JwtValidate {
            audience: String,
            issuer: String,
            jwks_url: String,
        },
        #[serde(rename = "mozilla")]
        Mozilla { required_groups: Vec<String> },
        #[serde(rename = "proxy_token")]
        ProxyToken {
            url: String,
            cache_secs: Option<u64>,
        },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    #[serde(deny_unknown_fields)]
    pub enum ServerAuth {
        #[serde(rename = "DANGEROUSLY_INSECURE")]
        Insecure,
        #[serde(rename = "jwt_hs256")]
        JwtHS256 { secret_key: String },
        #[serde(rename = "token")]
        Token { token: String },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Config {
        pub public_addr: SocketAddr,
        pub private_addr: Option<SocketAddr>,
        pub client_auth: ClientAuth,
        pub server_auth: ServerAuth,
    }

    pub fn from_path(conf_path: &Path) -> Result<Option<Config>> {
        super::try_read_config_file(&conf_path).context("Failed to load scheduler config file")
    }
}

#[cfg(feature = "dist-server")]
pub mod server {
    use super::HTTPUrl;
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};

    use crate::errors::*;

    const TEN_GIGS: u64 = 10 * 1024 * 1024 * 1024;
    fn default_toolchain_cache_size() -> u64 {
        TEN_GIGS
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    #[serde(deny_unknown_fields)]
    pub enum BuilderType {
        #[serde(rename = "docker")]
        Docker,
        #[serde(rename = "overlay")]
        Overlay {
            build_dir: PathBuf,
            bwrap_path: PathBuf,
        },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    #[serde(deny_unknown_fields)]
    pub enum SchedulerAuth {
        #[serde(rename = "DANGEROUSLY_INSECURE")]
        Insecure,
        #[serde(rename = "jwt_token")]
        JwtToken { token: String },
        #[serde(rename = "token")]
        Token { token: String },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Config {
        pub builder: BuilderType,
        pub cache_dir: PathBuf,
        pub public_addr: SocketAddr,
        pub private_addr: Option<SocketAddr>,
        pub scheduler_url: HTTPUrl,
        pub scheduler_auth: SchedulerAuth,
        #[serde(default = "default_toolchain_cache_size")]
        pub toolchain_cache_size: u64,
    }

    pub fn from_path(conf_path: &Path) -> Result<Option<Config>> {
        super::try_read_config_file(&conf_path).context("Failed to load server config file")
    }
}

#[test]
fn test_parse_size() {
    assert_eq!(None, parse_size(""));
    assert_eq!(None, parse_size("100"));
    assert_eq!(Some(2048), parse_size("2K"));
    assert_eq!(Some(10 * 1024 * 1024), parse_size("10M"));
    assert_eq!(Some(TEN_GIGS), parse_size("10G"));
    assert_eq!(Some(1024 * TEN_GIGS), parse_size("10T"));
}

#[test]
fn config_overrides() {
    let env_conf = EnvConfig {
        cache: CacheConfigs {
            azure: Some(AzureCacheConfig),
            disk: Some(DiskCacheConfig {
                dir: "/env-cache".into(),
                size: 5,
            }),
            redis: Some(RedisCacheConfig {
                url: "myotherredisurl".to_owned(),
            }),
            ..Default::default()
        },
    };

    let file_conf = FileConfig {
        cache: CacheConfigs {
            disk: Some(DiskCacheConfig {
                dir: "/file-cache".into(),
                size: 15,
            }),
            memcached: Some(MemcachedCacheConfig {
                url: "memurl".to_owned(),
            }),
            redis: Some(RedisCacheConfig {
                url: "myredisurl".to_owned(),
            }),
            ..Default::default()
        },
        dist: Default::default(),
    };

    assert_eq!(
        Config::from_env_and_file_configs(env_conf, file_conf),
        Config {
            caches: vec![
                CacheType::Redis(RedisCacheConfig {
                    url: "myotherredisurl".to_owned()
                }),
                CacheType::Memcached(MemcachedCacheConfig {
                    url: "memurl".to_owned()
                }),
                CacheType::Azure(AzureCacheConfig),
            ],
            fallback_cache: DiskCacheConfig {
                dir: "/env-cache".into(),
                size: 5,
            },
            dist: Default::default(),
        }
    );
}

#[test]
fn test_gcs_credentials_url() {
    env::set_var("SCCACHE_GCS_BUCKET", "my-bucket");
    env::set_var("SCCACHE_GCS_CREDENTIALS_URL", "http://localhost/");
    env::set_var("SCCACHE_GCS_RW_MODE", "READ_WRITE");

    let env_cfg = config_from_env();
    match env_cfg.cache.gcs {
        Some(GCSCacheConfig {
            ref bucket,
            ref url,
            rw_mode,
            ..
        }) => {
            assert_eq!(bucket, "my-bucket");
            match url {
                Some(ref url) => assert_eq!(url, "http://localhost/"),
                None => panic!("URL can't be none"),
            };
            assert_eq!(rw_mode, GCSCacheRWMode::ReadWrite);
        }
        None => unreachable!(),
    };
}
// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// For tokio_io::codec::length_delimited::Framed;
#![allow(deprecated)]

use crate::cache::{storage_from_config, Storage};
use crate::compiler::{
    get_compiler_info, CacheControl, CompileResult, Compiler, CompilerArguments, CompilerHasher,
    CompilerKind, CompilerProxy, DistType, MissType,
};
#[cfg(feature = "dist-client")]
use crate::config;
use crate::config::Config;
use crate::dist;
use crate::jobserver::Client;
use crate::mock_command::{CommandCreatorSync, ProcessCommandCreator};
use crate::protocol::{Compile, CompileFinished, CompileResponse, Request, Response};
use crate::util;
#[cfg(feature = "dist-client")]
use anyhow::Context as _;
use filetime::FileTime;
use futures::sync::mpsc;
use futures::{future, stream, Async, AsyncSink, Future, Poll, Sink, StartSend, Stream};
use futures_03::compat::Compat;
use futures_03::executor::ThreadPool;
use number_prefix::NumberPrefix;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::metadata;
use std::io::{self, Write};
#[cfg(feature = "dist-client")]
use std::mem;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::{ExitStatus, Output};
use std::rc::Rc;
use std::sync::Arc;
#[cfg(feature = "dist-client")]
use std::sync::Mutex;
use std::task::{Context, Waker};
use std::time::Duration;
use std::time::Instant;
use std::u64;
use tokio_compat::runtime::current_thread::Runtime;
use tokio_io::codec::length_delimited;
use tokio_io::codec::length_delimited::Framed;
use tokio_io::{AsyncRead, AsyncWrite};
use tokio_serde_bincode::{ReadBincode, WriteBincode};
use tokio_tcp::TcpListener;
use tokio_timer::{Delay, Timeout};
use tower::Service;

use crate::errors::*;

/// If the server is idle for this many seconds, shut down.
const DEFAULT_IDLE_TIMEOUT: u64 = 600;

/// If the dist client couldn't be created, retry creation at this number
/// of seconds from now (or later)
#[cfg(feature = "dist-client")]
const DIST_CLIENT_RECREATE_TIMEOUT: Duration = Duration::from_secs(30);

/// Result of background server startup.
#[derive(Debug, Serialize, Deserialize)]
pub enum ServerStartup {
    /// Server started successfully on `port`.
    Ok { port: u16 },
    /// Server Addr already in suse
    AddrInUse,
    /// Timed out waiting for server startup.
    TimedOut,
    /// Server encountered an error.
    Err { reason: String },
}

/// Get the time the server should idle for before shutting down.
fn get_idle_timeout() -> u64 {
    // A value of 0 disables idle shutdown entirely.
    env::var("SCCACHE_IDLE_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_IDLE_TIMEOUT)
}

fn notify_server_startup_internal<W: Write>(mut w: W, status: ServerStartup) -> Result<()> {
    util::write_length_prefixed_bincode(&mut w, status)
}

#[cfg(unix)]
fn notify_server_startup(name: &Option<OsString>, status: ServerStartup) -> Result<()> {
    use std::os::unix::net::UnixStream;
    let name = match *name {
        Some(ref s) => s,
        None => return Ok(()),
    };
    debug!("notify_server_startup({:?})", status);
    let stream = UnixStream::connect(name)?;
    notify_server_startup_internal(stream, status)
}

#[cfg(windows)]
fn notify_server_startup(name: &Option<OsString>, status: ServerStartup) -> Result<()> {
    use std::fs::OpenOptions;

    let name = match *name {
        Some(ref s) => s,
        None => return Ok(()),
    };
    let pipe = OpenOptions::new().write(true).read(true).open(name)?;
    notify_server_startup_internal(pipe, status)
}

#[cfg(unix)]
fn get_signal(status: ExitStatus) -> i32 {
    use std::os::unix::prelude::*;
    status.signal().expect("must have signal")
}
#[cfg(windows)]
fn get_signal(_status: ExitStatus) -> i32 {
    panic!("no signals on windows")
}

pub struct DistClientContainer {
    // The actual dist client state
    #[cfg(feature = "dist-client")]
    state: Mutex<DistClientState>,
}

#[cfg(feature = "dist-client")]
struct DistClientConfig {
    // Reusable items tied to an SccacheServer instance
    pool: ThreadPool,

    // From the static dist configuration
    scheduler_url: Option<config::HTTPUrl>,
    auth: config::DistAuth,
    cache_dir: PathBuf,
    toolchain_cache_size: u64,
    toolchains: Vec<config::DistToolchainConfig>,
    rewrite_includes_only: bool,
}

#[cfg(feature = "dist-client")]
enum DistClientState {
    #[cfg(feature = "dist-client")]
    Some(Box<DistClientConfig>, Arc<dyn dist::Client>),
    #[cfg(feature = "dist-client")]
    FailWithMessage(Box<DistClientConfig>, String),
    #[cfg(feature = "dist-client")]
    RetryCreateAt(Box<DistClientConfig>, Instant),
    Disabled,
}

#[cfg(not(feature = "dist-client"))]
impl DistClientContainer {
    #[cfg(not(feature = "dist-client"))]
    fn new(config: &Config, _: &ThreadPool) -> Self {
        if config.dist.scheduler_url.is_some() {
            warn!("Scheduler address configured but dist feature disabled, disabling distributed sccache")
        }
        Self {}
    }

    pub fn new_disabled() -> Self {
        Self {}
    }

    pub fn reset_state(&self) {}

    pub fn get_status(&self) -> DistInfo {
        DistInfo::Disabled("dist-client feature not selected".to_string())
    }

    fn get_client(&self) -> Result<Option<Arc<dyn dist::Client>>> {
        Ok(None)
    }
}

#[cfg(feature = "dist-client")]
impl DistClientContainer {
    fn new(config: &Config, pool: &ThreadPool) -> Self {
        let config = DistClientConfig {
            pool: pool.clone(),
            scheduler_url: config.dist.scheduler_url.clone(),
            auth: config.dist.auth.clone(),
            cache_dir: config.dist.cache_dir.clone(),
            toolchain_cache_size: config.dist.toolchain_cache_size,
            toolchains: config.dist.toolchains.clone(),
            rewrite_includes_only: config.dist.rewrite_includes_only,
        };
        let state = Self::create_state(config);
        Self {
            state: Mutex::new(state),
        }
    }

    pub fn new_disabled() -> Self {
        Self {
            state: Mutex::new(DistClientState::Disabled),
        }
    }

    pub fn reset_state(&self) {
        let mut guard = self.state.lock();
        let state = guard.as_mut().unwrap();
        let state: &mut DistClientState = &mut **state;
        match mem::replace(state, DistClientState::Disabled) {
            DistClientState::Some(cfg, _)
            | DistClientState::FailWithMessage(cfg, _)
            | DistClientState::RetryCreateAt(cfg, _) => {
                warn!("State reset. Will recreate");
                *state =
                    DistClientState::RetryCreateAt(cfg, Instant::now() - Duration::from_secs(1));
            }
            DistClientState::Disabled => (),
        }
    }

    pub fn get_status(&self) -> DistInfo {
        let mut guard = self.state.lock();
        let state = guard.as_mut().unwrap();
        let state: &mut DistClientState = &mut **state;
        match state {
            DistClientState::Disabled => DistInfo::Disabled("disabled".to_string()),
            DistClientState::FailWithMessage(cfg, _) => DistInfo::NotConnected(
                cfg.scheduler_url.clone(),
                "enabled, auth not configured".to_string(),
            ),
            DistClientState::RetryCreateAt(cfg, _) => DistInfo::NotConnected(
                cfg.scheduler_url.clone(),
                "enabled, not connected, will retry".to_string(),
            ),
            DistClientState::Some(cfg, client) => match client.do_get_status().wait() {
                Ok(res) => DistInfo::SchedulerStatus(cfg.scheduler_url.clone(), res),
                Err(_) => DistInfo::NotConnected(
                    cfg.scheduler_url.clone(),
                    "could not communicate with scheduler".to_string(),
                ),
            },
        }
    }

    fn get_client(&self) -> Result<Option<Arc<dyn dist::Client>>> {
        let mut guard = self.state.lock();
        let state = guard.as_mut().unwrap();
        let state: &mut DistClientState = &mut **state;
        Self::maybe_recreate_state(state);
        let res = match state {
            DistClientState::Some(_, dc) => Ok(Some(dc.clone())),
            DistClientState::Disabled | DistClientState::RetryCreateAt(_, _) => Ok(None),
            DistClientState::FailWithMessage(_, msg) => Err(anyhow!(msg.clone())),
        };
        if res.is_err() {
            let config = match mem::replace(state, DistClientState::Disabled) {
                DistClientState::FailWithMessage(config, _) => config,
                _ => unreachable!(),
            };
            // The client is most likely mis-configured, make sure we
            // re-create on our next attempt.
            *state =
                DistClientState::RetryCreateAt(config, Instant::now() - Duration::from_secs(1));
        }
        res
    }

    fn maybe_recreate_state(state: &mut DistClientState) {
        if let DistClientState::RetryCreateAt(_, instant) = *state {
            if instant > Instant::now() {
                return;
            }
            let config = match mem::replace(state, DistClientState::Disabled) {
                DistClientState::RetryCreateAt(config, _) => config,
                _ => unreachable!(),
            };
            info!("Attempting to recreate the dist client");
            *state = Self::create_state(*config)
        }
    }

    // Attempt to recreate the dist client
    fn create_state(config: DistClientConfig) -> DistClientState {
        macro_rules! try_or_retry_later {
            ($v:expr) => {{
                match $v {
                    Ok(v) => v,
                    Err(e) => {
                        // `{:?}` prints the full cause chain and backtrace.
                        error!("{:?}", e);
                        return DistClientState::RetryCreateAt(
                            Box::new(config),
                            Instant::now() + DIST_CLIENT_RECREATE_TIMEOUT,
                        );
                    }
                }
            }};
        }

        macro_rules! try_or_fail_with_message {
            ($v:expr) => {{
                match $v {
                    Ok(v) => v,
                    Err(e) => {
                        // `{:?}` prints the full cause chain and backtrace.
                        let errmsg = format!("{:?}", e);
                        error!("{}", errmsg);
                        return DistClientState::FailWithMessage(
                            Box::new(config),
                            errmsg.to_string(),
                        );
                    }
                }
            }};
        }
        match config.scheduler_url {
            Some(ref addr) => {
                let url = addr.to_url();
                info!("Enabling distributed sccache to {}", url);
                let auth_token = match &config.auth {
                    config::DistAuth::Token { token } => Ok(token.to_owned()),
                    config::DistAuth::Oauth2CodeGrantPKCE { auth_url, .. }
                    | config::DistAuth::Oauth2Implicit { auth_url, .. } => {
                        Self::get_cached_config_auth_token(auth_url)
                    }
                };
                let auth_token = try_or_fail_with_message!(auth_token
                    .context("could not load client auth token, run |sccache --dist-auth|"));
                let dist_client = dist::http::Client::new(
                    &config.pool,
                    url,
                    &config.cache_dir.join("client"),
                    config.toolchain_cache_size,
                    &config.toolchains,
                    auth_token,
                    config.rewrite_includes_only,
                );
                let dist_client =
                    try_or_retry_later!(dist_client.context("failure during dist client creation"));
                use crate::dist::Client;
                match dist_client.do_get_status().wait() {
                    Ok(res) => {
                        info!(
                            "Successfully created dist client with {:?} cores across {:?} servers",
                            res.num_cpus, res.num_servers
                        );
                        DistClientState::Some(Box::new(config), Arc::new(dist_client))
                    }
                    Err(_) => {
                        warn!("Scheduler address configured, but could not communicate with scheduler");
                        DistClientState::RetryCreateAt(
                            Box::new(config),
                            Instant::now() + DIST_CLIENT_RECREATE_TIMEOUT,
                        )
                    }
                }
            }
            None => {
                info!("No scheduler address configured, disabling distributed sccache");
                DistClientState::Disabled
            }
        }
    }

    fn get_cached_config_auth_token(auth_url: &str) -> Result<String> {
        let cached_config = config::CachedConfig::reload()?;
        cached_config
            .with(|c| c.dist.auth_tokens.get(auth_url).map(String::to_owned))
            .with_context(|| format!("token for url {} not present in cached config", auth_url))
    }
}

/// Start an sccache server, listening on `port`.
///
/// Spins an event loop handling client connections until a client
/// requests a shutdown.
pub fn start_server(config: &Config, port: u16) -> Result<()> {
    info!("start_server: port: {}", port);
    let client = unsafe { Client::new() };
    let runtime = Runtime::new()?;
    let pool = ThreadPool::builder()
        .pool_size(std::cmp::max(20, 2 * num_cpus::get()))
        .create()?;
    let dist_client = DistClientContainer::new(config, &pool);
    let storage = storage_from_config(config, &pool);
    let res = SccacheServer::<ProcessCommandCreator>::new(
        port,
        pool,
        runtime,
        client,
        dist_client,
        storage,
    );
    let notify = env::var_os("SCCACHE_STARTUP_NOTIFY");
    match res {
        Ok(srv) => {
            let port = srv.port();
            info!("server started, listening on port {}", port);
            notify_server_startup(&notify, ServerStartup::Ok { port })?;
            srv.run(future::empty::<(), ()>())?;
            Ok(())
        }
        Err(e) => {
            error!("failed to start server: {}", e);
            match e.downcast_ref::<io::Error>() {
                Some(io_err) if io::ErrorKind::AddrInUse == io_err.kind() => {
                    notify_server_startup(&notify, ServerStartup::AddrInUse)?;
                }
                _ => {
                    let reason = e.to_string();
                    notify_server_startup(&notify, ServerStartup::Err { reason })?;
                }
            };
            Err(e)
        }
    }
}

pub struct SccacheServer<C: CommandCreatorSync> {
    runtime: Runtime,
    listener: TcpListener,
    rx: mpsc::Receiver<ServerMessage>,
    timeout: Duration,
    service: SccacheService<C>,
    wait: WaitUntilZero,
}

impl<C: CommandCreatorSync> SccacheServer<C> {
    pub fn new(
        port: u16,
        pool: ThreadPool,
        runtime: Runtime,
        client: Client,
        dist_client: DistClientContainer,
        storage: Arc<dyn Storage>,
    ) -> Result<SccacheServer<C>> {
        let addr = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), port);
        let listener = TcpListener::bind(&SocketAddr::V4(addr))?;

        // Prepare the service which we'll use to service all incoming TCP
        // connections.
        let (tx, rx) = mpsc::channel(1);
        let (wait, info) = WaitUntilZero::new();
        let service = SccacheService::new(dist_client, storage, &client, pool, tx, info);

        Ok(SccacheServer {
            runtime,
            listener,
            rx,
            service,
            timeout: Duration::from_secs(get_idle_timeout()),
            wait,
        })
    }

    /// Configures how long this server will be idle before shutting down.
    #[allow(dead_code)]
    pub fn set_idle_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Set the storage this server will use.
    #[allow(dead_code)]
    pub fn set_storage(&mut self, storage: Arc<dyn Storage>) {
        self.service.storage = storage;
    }

    /// Returns a reference to a thread pool to run work on
    #[allow(dead_code)]
    pub fn pool(&self) -> &ThreadPool {
        &self.service.pool
    }

    /// Returns a reference to the command creator this server will use
    #[allow(dead_code)]
    pub fn command_creator(&self) -> &C {
        &self.service.creator
    }

    /// Returns the port that this server is bound to
    #[allow(dead_code)]
    pub fn port(&self) -> u16 {
        self.listener.local_addr().unwrap().port()
    }

    /// Runs this server to completion.
    ///
    /// If the `shutdown` future resolves then the server will be shut down,
    /// otherwise the server may naturally shut down if it becomes idle for too
    /// long anyway.
    pub fn run<F>(self, shutdown: F) -> io::Result<()>
    where
        F: Future,
    {
        self._run(Box::new(shutdown.then(|_| Ok(()))))
    }

    fn _run<'a>(self, shutdown: Box<dyn Future<Item = (), Error = ()> + 'a>) -> io::Result<()> {
        let SccacheServer {
            mut runtime,
            listener,
            rx,
            service,
            timeout,
            wait,
        } = self;

        // Create our "server future" which will simply handle all incoming
        // connections in separate tasks.
        let server = listener.incoming().for_each(move |socket| {
            trace!("incoming connection");
            tokio_compat::runtime::current_thread::TaskExecutor::current()
                .spawn_local(Box::new(service.clone().bind(socket).map_err(|err| {
                    error!("{}", err);
                })))
                .unwrap();
            Ok(())
        });

        // Right now there's a whole bunch of ways to shut down this server for
        // various purposes. These include:
        //
        // 1. The `shutdown` future above.
        // 2. An RPC indicating the server should shut down
        // 3. A period of inactivity (no requests serviced)
        //
        // These are all encapsulated wih the future that we're creating below.
        // The `ShutdownOrInactive` indicates the RPC or the period of
        // inactivity, and this is then select'd with the `shutdown` future
        // passed to this function.

        let shutdown = shutdown.map(|a| {
            info!("shutting down due to explicit signal");
            a
        });

        let mut futures = vec![
            Box::new(server) as Box<dyn Future<Item = _, Error = _>>,
            Box::new(
                shutdown
                    .map_err(|()| io::Error::new(io::ErrorKind::Other, "shutdown signal failed")),
            ),
        ];

        let shutdown_idle = ShutdownOrInactive {
            rx,
            timeout: if timeout != Duration::new(0, 0) {
                Some(Delay::new(Instant::now() + timeout))
            } else {
                None
            },
            timeout_dur: timeout,
        };
        futures.push(Box::new(shutdown_idle.map(|a| {
            info!("shutting down due to being idle or request");
            a
        })));

        let server = future::select_all(futures);
        runtime.block_on(server).map_err(|p| p.0)?;

        info!(
            "moving into the shutdown phase now, waiting at most 10 seconds \
             for all client requests to complete"
        );

        // Once our server has shut down either due to inactivity or a manual
        // request we still need to give a bit of time for all active
        // connections to finish. This `wait` future will resolve once all
        // instances of `SccacheService` have been dropped.
        //
        // Note that we cap the amount of time this can take, however, as we
        // don't want to wait *too* long.
        runtime
            .block_on(Timeout::new(Compat::new(wait), Duration::new(30, 0)))
            .map_err(|e| {
                if e.is_inner() {
                    e.into_inner().unwrap()
                } else {
                    io::Error::new(io::ErrorKind::Other, e)
                }
            })?;

        info!("ok, fully shutting down now");

        Ok(())
    }
}

type CompilerMap<C> = HashMap<PathBuf, Option<CompilerCacheEntry<C>>>;

/// entry of the compiler cache
struct CompilerCacheEntry<C: CommandCreatorSync> {
    /// compiler argument trait obj
    pub compiler: Box<dyn Compiler<C>>,
    /// modification time of the compilers executable file
    pub mtime: FileTime,
    /// distributed compilation extra info
    pub dist_info: Option<(PathBuf, FileTime)>,
}

impl<C> CompilerCacheEntry<C>
where
    C: CommandCreatorSync,
{
    fn new(
        compiler: Box<dyn Compiler<C>>,
        mtime: FileTime,
        dist_info: Option<(PathBuf, FileTime)>,
    ) -> Self {
        Self {
            compiler,
            mtime,
            dist_info,
        }
    }
}
/// Service implementation for sccache
#[derive(Clone)]
struct SccacheService<C: CommandCreatorSync> {
    /// Server statistics.
    stats: Rc<RefCell<ServerStats>>,

    /// Distributed sccache client
    dist_client: Rc<DistClientContainer>,

    /// Cache storage.
    storage: Arc<dyn Storage>,

    /// A cache of known compiler info.
    compilers: Rc<RefCell<CompilerMap<C>>>,

    /// map the cwd with compiler proxy path to a proxy resolver, which
    /// will dynamically resolve the input compiler for the current context
    /// (usually file or current working directory)
    /// the associated `FileTime` is the modification time of
    /// the compiler proxy, in order to track updates of the proxy itself
    compiler_proxies: Rc<RefCell<HashMap<PathBuf, (Box<dyn CompilerProxy<C>>, FileTime)>>>,

    /// Thread pool to execute work in
    pool: ThreadPool,

    /// An object for creating commands.
    ///
    /// This is mostly useful for unit testing, where we
    /// can mock this out.
    creator: C,

    /// Message channel used to learn about requests received by this server.
    ///
    /// Note that messages sent along this channel will keep the server alive
    /// (reset the idle timer) and this channel can also be used to shut down
    /// the entire server immediately via a message.
    tx: mpsc::Sender<ServerMessage>,

    /// Information tracking how many services (connected clients) are active.
    info: ActiveInfo,
}

type SccacheRequest = Message<Request, Body<()>>;
type SccacheResponse = Message<Response, Body<Response>>;

/// Messages sent from all services to the main event loop indicating activity.
///
/// Whenever a request is receive a `Request` message is sent which will reset
/// the idle shutdown timer, and otherwise a `Shutdown` message indicates that
/// a server shutdown was requested via an RPC.
pub enum ServerMessage {
    /// A message sent whenever a request is received.
    Request,
    /// Message sent whenever a shutdown request is received.
    Shutdown,
}

impl<C> Service<SccacheRequest> for SccacheService<C>
where
    C: CommandCreatorSync + 'static,
{
    type Response = SccacheResponse;
    type Error = Error;
    type Future = SFuture<Self::Response>;

    fn call(&mut self, req: SccacheRequest) -> Self::Future {
        trace!("handle_client");

        // Opportunistically let channel know that we've received a request. We
        // ignore failures here as well as backpressure as it's not imperative
        // that every message is received.
        drop(self.tx.clone().start_send(ServerMessage::Request));

        let res: SFuture<Response> = match req.into_inner() {
            Request::Compile(compile) => {
                debug!("handle_client: compile");
                self.stats.borrow_mut().compile_requests += 1;
                return self.handle_compile(compile);
            }
            Request::GetStats => {
                debug!("handle_client: get_stats");
                Box::new(self.get_info().map(|i| Response::Stats(Box::new(i))))
            }
            Request::DistStatus => {
                debug!("handle_client: dist_status");
                Box::new(self.get_dist_status().map(Response::DistStatus))
            }
            Request::ZeroStats => {
                debug!("handle_client: zero_stats");
                self.zero_stats();
                Box::new(self.get_info().map(|i| Response::Stats(Box::new(i))))
            }
            Request::Shutdown => {
                debug!("handle_client: shutdown");
                let future = self
                    .tx
                    .clone()
                    .send(ServerMessage::Shutdown)
                    .then(|_| Ok(()));
                let info_future = self.get_info();
                return Box::new(future.join(info_future).map(move |(_, info)| {
                    Message::WithoutBody(Response::ShuttingDown(Box::new(info)))
                }));
            }
        };

        Box::new(res.map(Message::WithoutBody))
    }

    fn poll_ready(&mut self) -> Poll<(), Self::Error> {
        Ok(Async::Ready(()))
    }
}

impl<C> SccacheService<C>
where
    C: CommandCreatorSync,
{
    pub fn new(
        dist_client: DistClientContainer,
        storage: Arc<dyn Storage>,
        client: &Client,
        pool: ThreadPool,
        tx: mpsc::Sender<ServerMessage>,
        info: ActiveInfo,
    ) -> SccacheService<C> {
        SccacheService {
            stats: Rc::new(RefCell::new(ServerStats::default())),
            dist_client: Rc::new(dist_client),
            storage,
            compilers: Rc::new(RefCell::new(HashMap::new())),
            compiler_proxies: Rc::new(RefCell::new(HashMap::new())),
            pool,
            creator: C::new(client),
            tx,
            info,
        }
    }

    fn bind<T>(mut self, socket: T) -> impl Future<Item = (), Error = Error>
    where
        T: AsyncRead + AsyncWrite + 'static,
    {
        let mut builder = length_delimited::Builder::new();
        if let Ok(max_frame_length_str) = env::var("SCCACHE_MAX_FRAME_LENGTH") {
            if let Ok(max_frame_length) = max_frame_length_str.parse::<usize>() {
                builder.max_frame_length(max_frame_length);
            } else {
                warn!("Content of SCCACHE_MAX_FRAME_LENGTH is  not a valid number, using default");
            }
        }
        let io = builder.new_framed(socket);

        let (sink, stream) = SccacheTransport {
            inner: WriteBincode::new(ReadBincode::new(io)),
        }
        .split();
        let sink = sink.sink_from_err::<Error>();

        stream
            .from_err::<Error>()
            .and_then(move |input| self.call(input))
            .and_then(|message| {
                let f: Box<dyn Stream<Item = _, Error = _>> = match message {
                    Message::WithoutBody(message) => Box::new(stream::once(Ok(Frame::Message {
                        message,
                        body: false,
                    }))),
                    Message::WithBody(message, body) => Box::new(
                        stream::once(Ok(Frame::Message {
                            message,
                            body: true,
                        }))
                        .chain(Compat::new(body).map(|chunk| Frame::Body { chunk: Some(chunk) }))
                        .chain(stream::once(Ok(Frame::Body { chunk: None }))),
                    ),
                };
                Ok(f.from_err::<Error>())
            })
            .flatten()
            .forward(sink)
            .map(|_| ())
    }

    /// Get dist status.
    fn get_dist_status(&self) -> SFuture<DistInfo> {
        f_ok(self.dist_client.get_status())
    }

    /// Get info and stats about the cache.
    fn get_info(&self) -> SFuture<ServerInfo> {
        let stats = self.stats.borrow().clone();
        let cache_location = self.storage.location();
        Box::new(
            self.storage
                .current_size()
                .join(self.storage.max_size())
                .map(move |(cache_size, max_cache_size)| ServerInfo {
                    stats,
                    cache_location,
                    cache_size,
                    max_cache_size,
                }),
        )
    }

    /// Zero stats about the cache.
    fn zero_stats(&self) {
        *self.stats.borrow_mut() = ServerStats::default();
    }

    /// Handle a compile request from a client.
    ///
    /// This will handle a compile request entirely, generating a response with
    /// the inital information and an optional body which will eventually
    /// contain the results of the compilation.
    fn handle_compile(&self, compile: Compile) -> SFuture<SccacheResponse> {
        let exe = compile.exe;
        let cmd = compile.args;
        let cwd: PathBuf = compile.cwd.into();
        let env_vars = compile.env_vars;
        let me = self.clone();

        Box::new(
            self.compiler_info(exe.into(), cwd.clone(), &env_vars)
                .map(move |info| me.check_compiler(info, cmd, cwd, env_vars)),
        )
    }

    /// Look up compiler info from the cache for the compiler `path`.
    /// If not cached, determine the compiler type and cache the result.
    fn compiler_info(
        &self,
        path: PathBuf,
        cwd: PathBuf,
        env: &[(OsString, OsString)],
    ) -> SFuture<Result<Box<dyn Compiler<C>>>> {
        trace!("compiler_info");

        let me = self.clone();
        let me1 = self.clone();

        // lookup if compiler proxy exists for the current compiler path

        let path2 = path.clone();
        let path1 = path.clone();
        let env = env.to_vec();

        let resolve_w_proxy = {
            let compiler_proxies_borrow = self.compiler_proxies.borrow();

            if let Some((compiler_proxy, _filetime)) = compiler_proxies_borrow.get(&path) {
                let fut = compiler_proxy.resolve_proxied_executable(
                    self.creator.clone(),
                    cwd.clone(),
                    env.as_slice(),
                );
                Box::new(fut.then(|res: Result<_>| Ok(res.ok())))
            } else {
                f_ok(None)
            }
        };

        // use the supplied compiler path as fallback, lookup its modification time too
        let w_fallback = resolve_w_proxy.then(move |res: Result<Option<(PathBuf, FileTime)>>| {
            let opt = match res {
                Ok(Some(x)) => Some(x), // TODO resolve the path right away
                _ => {
                    // fallback to using the path directly
                    metadata(&path2)
                        .map(|attr| FileTime::from_last_modification_time(&attr))
                        .ok()
                        .map(move |filetime| (path2, filetime))
                }
            };
            f_ok(opt)
        });

        let lookup_compiler = w_fallback.and_then(move |opt: Option<(PathBuf, FileTime)>| {
            let (resolved_compiler_path, mtime) =
                opt.expect("Must contain sane data, otherwise mtime is not avail");

            let dist_info = match me1.dist_client.get_client() {
                Ok(Some(ref client)) => {
                    if let Some(archive) = client.get_custom_toolchain(&resolved_compiler_path) {
                        match metadata(&archive)
                            .map(|attr| FileTime::from_last_modification_time(&attr))
                        {
                            Ok(mtime) => Some((archive, mtime)),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let opt = match me1.compilers.borrow().get(&resolved_compiler_path) {
                // It's a hit only if the mtime and dist archive data matches.
                Some(&Some(ref entry)) => {
                    if entry.mtime == mtime && entry.dist_info == dist_info {
                        Some(entry.compiler.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            };
            f_ok((resolved_compiler_path, mtime, opt, dist_info))
        });

        let obtain = lookup_compiler.and_then(
            move |(resolved_compiler_path, mtime, opt, dist_info): (
                PathBuf,
                FileTime,
                Option<Box<dyn Compiler<C>>>,
                Option<(PathBuf, FileTime)>,
            )| {
                match opt {
                    Some(info) => {
                        trace!("compiler_info cache hit");
                        f_ok(Ok(info))
                    }
                    None => {
                        trace!("compiler_info cache miss");
                        // Check the compiler type and return the result when
                        // finished. This generally involves invoking the compiler,
                        // so do it asynchronously.

                        // the compiler path might be compiler proxy, so it is important to use
                        // `path` (or its clone `path1`) to resolve using that one, not using `resolved_compiler_path`
                        let x = get_compiler_info::<C>(
                            me.creator.clone(),
                            &path1,
                            &cwd,
                            env.as_slice(),
                            &me.pool,
                            dist_info.clone().map(|(p, _)| p),
                        );

                        Box::new(x.then(
                            move |info: Result<(
                                Box<dyn Compiler<C>>,
                                Option<Box<dyn CompilerProxy<C>>>,
                            )>| {
                                match info {
                                    Ok((ref c, ref proxy)) => {
                                        // register the proxy for this compiler, so it will be used directly from now on
                                        // and the true/resolved compiler will create table hits in the hash map
                                        // based on the resolved path
                                        if let Some(proxy) = proxy {
                                            trace!(
                                                "Inserting new path proxy {:?} @ {:?} -> {:?}",
                                                &path,
                                                &cwd,
                                                resolved_compiler_path
                                            );
                                            let proxy: Box<dyn CompilerProxy<C>> =
                                                proxy.box_clone();
                                            me.compiler_proxies
                                                .borrow_mut()
                                                .insert(path, (proxy, mtime));
                                        }
                                        // TODO add some safety checks in case a proxy exists, that the initial `path` is not
                                        // TODO the same as the resolved compiler binary

                                        // cache
                                        let map_info =
                                            CompilerCacheEntry::new(c.clone(), mtime, dist_info);
                                        trace!(
                                            "Inserting POSSIBLY PROXIED cache map info for {:?}",
                                            &resolved_compiler_path
                                        );
                                        me.compilers
                                            .borrow_mut()
                                            .insert(resolved_compiler_path, Some(map_info));
                                    }
                                    Err(_) => {
                                        trace!("Inserting PLAIN cache map info for {:?}", &path);
                                        me.compilers.borrow_mut().insert(path, None);
                                    }
                                }
                                // drop the proxy information, response is compiler only
                                let r: Result<Box<dyn Compiler<C>>> = info.map(|info| info.0);
                                f_ok(r)
                            },
                        ))
                    }
                }
            },
        );

        Box::new(obtain)
    }

    /// Check that we can handle and cache `cmd` when run with `compiler`.
    /// If so, run `start_compile_task` to execute it.
    fn check_compiler(
        &self,
        compiler: Result<Box<dyn Compiler<C>>>,
        cmd: Vec<OsString>,
        cwd: PathBuf,
        env_vars: Vec<(OsString, OsString)>,
    ) -> SccacheResponse {
        let mut stats = self.stats.borrow_mut();
        match compiler {
            Err(e) => {
                debug!("check_compiler: Unsupported compiler: {}", e.to_string());
                stats.requests_unsupported_compiler += 1;
                return Message::WithoutBody(Response::Compile(
                    CompileResponse::UnsupportedCompiler(OsString::from(e.to_string())),
                ));
            }
            Ok(c) => {
                debug!("check_compiler: Supported compiler");
                // Now check that we can handle this compiler with
                // the provided commandline.
                match c.parse_arguments(&cmd, &cwd) {
                    CompilerArguments::Ok(hasher) => {
                        debug!("parse_arguments: Ok: {:?}", cmd);
                        stats.requests_executed += 1;
                        let (tx, rx) = Body::pair();
                        self.start_compile_task(c, hasher, cmd, cwd, env_vars, tx);
                        let res = CompileResponse::CompileStarted;
                        return Message::WithBody(Response::Compile(res), rx);
                    }
                    CompilerArguments::CannotCache(why, extra_info) => {
                        if let Some(extra_info) = extra_info {
                            debug!(
                                "parse_arguments: CannotCache({}, {}): {:?}",
                                why, extra_info, cmd
                            )
                        } else {
                            debug!("parse_arguments: CannotCache({}): {:?}", why, cmd)
                        }
                        stats.requests_not_cacheable += 1;
                        *stats.not_cached.entry(why.to_string()).or_insert(0) += 1;
                    }
                    CompilerArguments::NotCompilation => {
                        debug!("parse_arguments: NotCompilation: {:?}", cmd);
                        stats.requests_not_compile += 1;
                    }
                }
            }
        }

        let res = CompileResponse::UnhandledCompile;
        Message::WithoutBody(Response::Compile(res))
    }

    /// Given compiler arguments `arguments`, look up
    /// a compile result in the cache or execute the compilation and store
    /// the result in the cache.
    fn start_compile_task(
        &self,
        compiler: Box<dyn Compiler<C>>,
        hasher: Box<dyn CompilerHasher<C>>,
        arguments: Vec<OsString>,
        cwd: PathBuf,
        env_vars: Vec<(OsString, OsString)>,
        tx: mpsc::Sender<Result<Response>>,
    ) {
        let force_recache = env_vars
            .iter()
            .any(|&(ref k, ref _v)| k.as_os_str() == OsStr::new("SCCACHE_RECACHE"));
        let cache_control = if force_recache {
            CacheControl::ForceRecache
        } else {
            CacheControl::Default
        };
        let out_pretty = hasher.output_pretty().into_owned();
        let color_mode = hasher.color_mode();
        let result = hasher.get_cached_or_compile(
            self.dist_client.get_client(),
            self.creator.clone(),
            self.storage.clone(),
            arguments,
            cwd,
            env_vars,
            cache_control,
            self.pool.clone(),
        );
        let me = self.clone();
        let kind = compiler.kind();
        let task = result.then(move |result| {
            let mut cache_write = None;
            let mut stats = me.stats.borrow_mut();
            let mut res = CompileFinished {
                color_mode,
                ..Default::default()
            };
            match result {
                Ok((compiled, out)) => {
                    match compiled {
                        CompileResult::Error => {
                            stats.cache_errors.increment(&kind);
                        }
                        CompileResult::CacheHit(duration) => {
                            stats.cache_hits.increment(&kind);
                            stats.cache_read_hit_duration += duration;
                        }
                        CompileResult::CacheMiss(miss_type, dist_type, duration, future) => {
                            match dist_type {
                                DistType::NoDist => {}
                                DistType::Ok(server_addresses) => {
                                    let server = server_addresses.server_public_id().addr().to_string();
                                    let server_count =
                                        stats.dist_compiles.entry(server).or_insert(0);
                                    *server_count += 1;
                                }
                                DistType::Error => stats.dist_errors += 1,
                            }
                            match miss_type {
                                MissType::Normal => {}
                                MissType::ForcedRecache => {
                                    stats.forced_recaches += 1;
                                }
                                MissType::TimedOut => {
                                    stats.cache_timeouts += 1;
                                }
                                MissType::CacheReadError => {
                                    stats.cache_errors.increment(&kind);
                                }
                            }
                            stats.cache_misses.increment(&kind);
                            stats.cache_read_miss_duration += duration;
                            cache_write = Some(future);
                        }
                        CompileResult::NotCacheable => {
                            stats.cache_misses.increment(&kind);
                            stats.non_cacheable_compilations += 1;
                        }
                        CompileResult::CompileFailed => {
                            stats.compile_fails += 1;
                        }
                    };
                    let Output {
                        status,
                        stdout,
                        stderr,
                    } = out;
                    trace!("CompileFinished retcode: {}", status);
                    match status.code() {
                        Some(code) => res.retcode = Some(code),
                        None => res.signal = Some(get_signal(status)),
                    };
                    res.stdout = stdout;
                    res.stderr = stderr;
                }
                Err(err) => {
                    match err.downcast::<ProcessError>() {
                        Ok(ProcessError(output)) => {
                            debug!("Compilation failed: {:?}", output);
                            stats.compile_fails += 1;
                            match output.status.code() {
                                Some(code) => res.retcode = Some(code),
                                None => res.signal = Some(get_signal(output.status)),
                            };
                            res.stdout = output.stdout;
                            res.stderr = output.stderr;
                        }
                        Err(err) => match err.downcast::<HttpClientError>() {
                            Ok(HttpClientError(msg)) => {
                                me.dist_client.reset_state();
                                let errmsg =
                                    format!("[{:?}] http error status: {}", out_pretty, msg);
                                error!("{}", errmsg);
                                res.retcode = Some(1);
                                res.stderr = errmsg.as_bytes().to_vec();
                            }
                            Err(err) => {
                                use std::fmt::Write;

                                error!("[{:?}] fatal error: {}", out_pretty, err);

                                let mut error = "sccache: encountered fatal error\n".to_string();
                                let _ = writeln!(error, "sccache: error: {}", err);
                                for e in err.chain() {
                                    error!("[{:?}] \t{}", out_pretty, e);
                                    let _ = writeln!(error, "sccache: caused by: {}", e);
                                }
                                stats.cache_errors.increment(&kind);
                                //TODO: figure out a better way to communicate this?
                                res.retcode = Some(-2);
                                res.stderr = error.into_bytes();
                            }
                        },
                    }
                }
            };
            let send = tx.send(Ok(Response::CompileFinished(res)));

            let me = me.clone();
            let cache_write = cache_write.then(move |result| {
                match result {
                    Err(e) => {
                        debug!("Error executing cache write: {}", e);
                        me.stats.borrow_mut().cache_write_errors += 1;
                    }
                    //TODO: save cache stats!
                    Ok(Some(info)) => {
                        debug!(
                            "[{}]: Cache write finished in {}",
                            info.object_file_pretty,
                            util::fmt_duration_as_secs(&info.duration)
                        );
                        me.stats.borrow_mut().cache_writes += 1;
                        me.stats.borrow_mut().cache_write_duration += info.duration;
                    }

                    Ok(None) => {}
                }
                Ok(())
            });

            send.join(cache_write).then(|_| Ok(()))
        });

        tokio_compat::runtime::current_thread::TaskExecutor::current()
            .spawn_local(Box::new(task))
            .unwrap();
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PerLanguageCount {
    counts: HashMap<String, u64>,
}

impl PerLanguageCount {
    fn increment(&mut self, kind: &CompilerKind) {
        let key = kind.lang_kind();
        let count = self.counts.entry(key).or_insert(0);
        *count += 1;
    }

    pub fn all(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn get(&self, key: &str) -> Option<&u64> {
        self.counts.get(key)
    }

    pub fn new() -> PerLanguageCount {
        Self::default()
    }
}

/// Statistics about the server.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ServerStats {
    /// The count of client compile requests.
    pub compile_requests: u64,
    /// The count of client requests that used an unsupported compiler.
    pub requests_unsupported_compiler: u64,
    /// The count of client requests that were not compilation.
    pub requests_not_compile: u64,
    /// The count of client requests that were not cacheable.
    pub requests_not_cacheable: u64,
    /// The count of client requests that were executed.
    pub requests_executed: u64,
    /// The count of errors handling compile requests (per language).
    pub cache_errors: PerLanguageCount,
    /// The count of cache hits for handled compile requests (per language).
    pub cache_hits: PerLanguageCount,
    /// The count of cache misses for handled compile requests (per language).
    pub cache_misses: PerLanguageCount,
    /// The count of cache misses because the cache took too long to respond.
    pub cache_timeouts: u64,
    /// The count of errors reading cache entries.
    pub cache_read_errors: u64,
    /// The count of compilations which were successful but couldn't be cached.
    pub non_cacheable_compilations: u64,
    /// The count of compilations which forcibly ignored the cache.
    pub forced_recaches: u64,
    /// The count of errors writing to cache.
    pub cache_write_errors: u64,
    /// The number of successful cache writes.
    pub cache_writes: u64,
    /// The total time spent writing cache entries.
    pub cache_write_duration: Duration,
    /// The total time spent reading cache hits.
    pub cache_read_hit_duration: Duration,
    /// The total time spent reading cache misses.
    pub cache_read_miss_duration: Duration,
    /// The count of compilation failures.
    pub compile_fails: u64,
    /// Counts of reasons why compiles were not cached.
    pub not_cached: HashMap<String, usize>,
    /// The count of compilations that were successfully distributed indexed
    /// by the server that ran those compilations.
    pub dist_compiles: HashMap<String, usize>,
    /// The count of compilations that were distributed but failed and had to be re-run locally
    pub dist_errors: u64,
}

/// Info and stats about the server.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ServerInfo {
    pub stats: ServerStats,
    pub cache_location: String,
    pub cache_size: Option<u64>,
    pub max_cache_size: Option<u64>,
}

/// Status of the dist client.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DistInfo {
    Disabled(String),
    #[cfg(feature = "dist-client")]
    NotConnected(Option<config::HTTPUrl>, String),
    #[cfg(feature = "dist-client")]
    SchedulerStatus(Option<config::HTTPUrl>, dist::SchedulerStatusResult),
}

impl Default for ServerStats {
    fn default() -> ServerStats {
        ServerStats {
            compile_requests: u64::default(),
            requests_unsupported_compiler: u64::default(),
            requests_not_compile: u64::default(),
            requests_not_cacheable: u64::default(),
            requests_executed: u64::default(),
            cache_errors: PerLanguageCount::new(),
            cache_hits: PerLanguageCount::new(),
            cache_misses: PerLanguageCount::new(),
            cache_timeouts: u64::default(),
            cache_read_errors: u64::default(),
            non_cacheable_compilations: u64::default(),
            forced_recaches: u64::default(),
            cache_write_errors: u64::default(),
            cache_writes: u64::default(),
            cache_write_duration: Duration::new(0, 0),
            cache_read_hit_duration: Duration::new(0, 0),
            cache_read_miss_duration: Duration::new(0, 0),
            compile_fails: u64::default(),
            not_cached: HashMap::new(),
            dist_compiles: HashMap::new(),
            dist_errors: u64::default(),
        }
    }
}

impl ServerStats {
    /// Print stats to stdout in a human-readable format.
    ///
    /// Return the formatted width of each of the (name, value) columns.
    fn print(&self) -> (usize, usize) {
        macro_rules! set_stat {
            ($vec:ident, $var:expr, $name:expr) => {{
                // name, value, suffix length
                $vec.push(($name.to_string(), $var.to_string(), 0));
            }};
        }

        macro_rules! set_lang_stat {
            ($vec:ident, $var:expr, $name:expr) => {{
                $vec.push(($name.to_string(), $var.all().to_string(), 0));
                let mut sorted_stats: Vec<_> = $var.counts.iter().collect();
                sorted_stats.sort_by_key(|v| v.0);
                for (lang, count) in sorted_stats.iter() {
                    $vec.push((format!("{} ({})", $name, lang), count.to_string(), 0));
                }
            }};
        }

        macro_rules! set_duration_stat {
            ($vec:ident, $dur:expr, $num:expr, $name:expr) => {{
                let s = if $num > 0 {
                    $dur / $num as u32
                } else {
                    Default::default()
                };
                // name, value, suffix length
                $vec.push(($name.to_string(), util::fmt_duration_as_secs(&s), 2));
            }};
        }

        let mut stats_vec = vec![];
        //TODO: this would be nice to replace with a custom derive implementation.
        set_stat!(stats_vec, self.compile_requests, "Compile requests");
        set_stat!(
            stats_vec,
            self.requests_executed,
            "Compile requests executed"
        );
        set_lang_stat!(stats_vec, self.cache_hits, "Cache hits");
        set_lang_stat!(stats_vec, self.cache_misses, "Cache misses");
        set_stat!(stats_vec, self.cache_timeouts, "Cache timeouts");
        set_stat!(stats_vec, self.cache_read_errors, "Cache read errors");
        set_stat!(stats_vec, self.forced_recaches, "Forced recaches");
        set_stat!(stats_vec, self.cache_write_errors, "Cache write errors");
        set_stat!(stats_vec, self.compile_fails, "Compilation failures");
        set_lang_stat!(stats_vec, self.cache_errors, "Cache errors");
        set_stat!(
            stats_vec,
            self.non_cacheable_compilations,
            "Non-cacheable compilations"
        );
        set_stat!(
            stats_vec,
            self.requests_not_cacheable,
            "Non-cacheable calls"
        );
        set_stat!(
            stats_vec,
            self.requests_not_compile,
            "Non-compilation calls"
        );
        set_stat!(
            stats_vec,
            self.requests_unsupported_compiler,
            "Unsupported compiler calls"
        );
        set_duration_stat!(
            stats_vec,
            self.cache_write_duration,
            self.cache_writes,
            "Average cache write"
        );
        set_duration_stat!(
            stats_vec,
            self.cache_read_miss_duration,
            self.cache_misses.all(),
            "Average cache read miss"
        );
        set_duration_stat!(
            stats_vec,
            self.cache_read_hit_duration,
            self.cache_hits.all(),
            "Average cache read hit"
        );
        set_stat!(
            stats_vec,
            self.dist_errors,
            "Failed distributed compilations"
        );
        let name_width = stats_vec
            .iter()
            .map(|&(ref n, _, _)| n.len())
            .max()
            .unwrap();
        let stat_width = stats_vec
            .iter()
            .map(|&(_, ref s, _)| s.len())
            .max()
            .unwrap();
        for (name, stat, suffix_len) in stats_vec {
            println!(
                "{:<name_width$} {:>stat_width$}",
                name,
                stat,
                name_width = name_width,
                stat_width = stat_width + suffix_len
            );
        }
        if !self.dist_compiles.is_empty() {
            println!("\nSuccessful distributed compiles");
            let mut counts: Vec<_> = self.dist_compiles.iter().collect();
            counts.sort_by(|(_, c1), (_, c2)| c1.cmp(c2).reverse());
            for (reason, count) in counts {
                println!(
                    "  {:<name_width$} {:>stat_width$}",
                    reason,
                    count,
                    name_width = name_width - 2,
                    stat_width = stat_width
                );
            }
        }
        if !self.not_cached.is_empty() {
            println!("\nNon-cacheable reasons:");
            let mut counts: Vec<_> = self.not_cached.iter().collect();
            counts.sort_by(|(_, c1), (_, c2)| c1.cmp(c2).reverse());
            for (reason, count) in counts {
                println!(
                    "{:<name_width$} {:>stat_width$}",
                    reason,
                    count,
                    name_width = name_width,
                    stat_width = stat_width
                );
            }
            println!();
        }
        (name_width, stat_width)
    }
}

impl ServerInfo {
    /// Print info to stdout in a human-readable format.
    pub fn print(&self) {
        let (name_width, stat_width) = self.stats.print();
        println!(
            "{:<name_width$} {}",
            "Cache location",
            self.cache_location,
            name_width = name_width
        );
        for &(name, val) in &[
            ("Cache size", &self.cache_size),
            ("Max cache size", &self.max_cache_size),
        ] {
            if let Some(val) = *val {
                let (val, suffix) = match NumberPrefix::binary(val as f64) {
                    NumberPrefix::Standalone(bytes) => (bytes.to_string(), "bytes".to_string()),
                    NumberPrefix::Prefixed(prefix, n) => {
                        (format!("{:.0}", n), format!("{}B", prefix))
                    }
                };
                println!(
                    "{:<name_width$} {:>stat_width$} {}",
                    name,
                    val,
                    suffix,
                    name_width = name_width,
                    stat_width = stat_width
                );
            }
        }
    }
}

enum Frame<R, R1> {
    Body { chunk: Option<R1> },
    Message { message: R, body: bool },
}

struct Body<R> {
    receiver: mpsc::Receiver<Result<R>>,
}

impl<R> Body<R> {
    fn pair() -> (mpsc::Sender<Result<R>>, Self) {
        let (tx, rx) = mpsc::channel(0);
        (tx, Body { receiver: rx })
    }
}

impl<R> futures_03::Stream for Body<R> {
    type Item = Result<R>;
    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match Pin::new(&mut self.receiver).poll().unwrap() {
            Async::Ready(item) => std::task::Poll::Ready(item),
            Async::NotReady => std::task::Poll::Pending,
        }
    }
}

enum Message<R, B> {
    WithBody(R, B),
    WithoutBody(R),
}

impl<R, B> Message<R, B> {
    fn into_inner(self) -> R {
        match self {
            Message::WithBody(r, _) => r,
            Message::WithoutBody(r) => r,
        }
    }
}

/// Implementation of `Stream + Sink` that tokio-proto is expecting
///
/// This type is composed of a few layers:
///
/// * First there's `I`, the I/O object implementing `AsyncRead` and
///   `AsyncWrite`
/// * Next that's framed using the `length_delimited` module in tokio-io giving
///   us a `Sink` and `Stream` of `BytesMut`.
/// * Next that sink/stream is wrapped in `ReadBincode` which will cause the
///   `Stream` implementation to switch from `BytesMut` to `Request` by parsing
///   the bytes  bincode.
/// * Finally that sink/stream is wrapped in `WriteBincode` which will cause the
///   `Sink` implementation to switch from `BytesMut` to `Response` meaning that
///   all `Response` types pushed in will be converted to `BytesMut` and pushed
///   below.
struct SccacheTransport<I: AsyncRead + AsyncWrite> {
    inner: WriteBincode<ReadBincode<Framed<I>, Request>, Response>,
}

impl<I: AsyncRead + AsyncWrite> Stream for SccacheTransport<I> {
    type Item = Message<Request, Body<()>>;
    type Error = io::Error;

    fn poll(&mut self) -> Poll<Option<Self::Item>, io::Error> {
        let msg = try_ready!(self.inner.poll().map_err(|e| {
            error!("SccacheTransport::poll failed: {}", e);
            io::Error::new(io::ErrorKind::Other, e)
        }));
        Ok(msg.map(Message::WithoutBody).into())
    }
}

impl<I: AsyncRead + AsyncWrite> Sink for SccacheTransport<I> {
    type SinkItem = Frame<Response, Response>;
    type SinkError = io::Error;

    fn start_send(&mut self, item: Self::SinkItem) -> StartSend<Self::SinkItem, io::Error> {
        match item {
            Frame::Message { message, body } => match self.inner.start_send(message)? {
                AsyncSink::Ready => Ok(AsyncSink::Ready),
                AsyncSink::NotReady(message) => {
                    Ok(AsyncSink::NotReady(Frame::Message { message, body }))
                }
            },
            Frame::Body { chunk: Some(chunk) } => match self.inner.start_send(chunk)? {
                AsyncSink::Ready => Ok(AsyncSink::Ready),
                AsyncSink::NotReady(chunk) => {
                    Ok(AsyncSink::NotReady(Frame::Body { chunk: Some(chunk) }))
                }
            },
            Frame::Body { chunk: None } => Ok(AsyncSink::Ready),
        }
    }

    fn poll_complete(&mut self) -> Poll<(), io::Error> {
        self.inner.poll_complete()
    }

    fn close(&mut self) -> Poll<(), io::Error> {
        self.inner.close()
    }
}

struct ShutdownOrInactive {
    rx: mpsc::Receiver<ServerMessage>,
    timeout: Option<Delay>,
    timeout_dur: Duration,
}

impl Future for ShutdownOrInactive {
    type Item = ();
    type Error = io::Error;

    fn poll(&mut self) -> Poll<(), io::Error> {
        loop {
            match self.rx.poll().unwrap() {
                Async::NotReady => break,
                // Shutdown received!
                Async::Ready(Some(ServerMessage::Shutdown)) => return Ok(().into()),
                Async::Ready(Some(ServerMessage::Request)) => {
                    if self.timeout_dur != Duration::new(0, 0) {
                        self.timeout = Some(Delay::new(Instant::now() + self.timeout_dur));
                    }
                }
                // All services have shut down, in theory this isn't possible...
                Async::Ready(None) => return Ok(().into()),
            }
        }
        match self.timeout {
            None => Ok(Async::NotReady),
            Some(ref mut timeout) => timeout
                .poll()
                .map_err(|err| io::Error::new(io::ErrorKind::Other, err)),
        }
    }
}

/// Helper future which tracks the `ActiveInfo` below. This future will resolve
/// once all instances of `ActiveInfo` have been dropped.
struct WaitUntilZero {
    info: Rc<RefCell<Info>>,
}

struct ActiveInfo {
    info: Rc<RefCell<Info>>,
}

struct Info {
    active: usize,
    waker: Option<Waker>,
}

impl WaitUntilZero {
    fn new() -> (WaitUntilZero, ActiveInfo) {
        let info = Rc::new(RefCell::new(Info {
            active: 1,
            waker: None,
        }));

        (WaitUntilZero { info: info.clone() }, ActiveInfo { info })
    }
}

impl Clone for ActiveInfo {
    fn clone(&self) -> ActiveInfo {
        self.info.borrow_mut().active += 1;
        ActiveInfo {
            info: self.info.clone(),
        }
    }
}

impl Drop for ActiveInfo {
    fn drop(&mut self) {
        let mut info = self.info.borrow_mut();
        info.active -= 1;
        if info.active == 0 {
            if let Some(waker) = info.waker.take() {
                waker.wake();
            }
        }
    }
}

impl std::future::Future for WaitUntilZero {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> std::task::Poll<Self::Output> {
        let mut info = self.info.borrow_mut();
        if info.active == 0 {
            std::task::Poll::Ready(Ok(()))
        } else {
            info.waker = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}
#![cfg(all(feature = "dist-client", feature = "dist-server"))]

extern crate assert_cmd;
#[macro_use]
extern crate log;
extern crate sccache;
extern crate serde_json;

use crate::harness::{
    get_stats, sccache_command, start_local_daemon, stop_local_daemon, write_json_cfg, write_source,
};
use assert_cmd::prelude::*;
use sccache::config::HTTPUrl;
use sccache::dist::{
    AssignJobResult, CompileCommand, InputsReader, JobId, JobState, RunJobResult, ServerIncoming,
    ServerOutgoing, SubmitToolchainResult, Toolchain, ToolchainReader,
};
use std::ffi::OsStr;
use std::path::Path;

use sccache::errors::*;

mod harness;

fn basic_compile(tmpdir: &Path, sccache_cfg_path: &Path, sccache_cached_cfg_path: &Path) {
    let envs: Vec<(_, &OsStr)> = vec![
        ("RUST_BACKTRACE", "1".as_ref()),
        ("RUST_LOG", "sccache=trace".as_ref()),
        ("SCCACHE_CONF", sccache_cfg_path.as_ref()),
        ("SCCACHE_CACHED_CONF", sccache_cached_cfg_path.as_ref()),
    ];
    let source_file = "x.c";
    let obj_file = "x.o";
    write_source(tmpdir, source_file, "#if !defined(SCCACHE_TEST_DEFINE)\n#error SCCACHE_TEST_DEFINE is not defined\n#endif\nint x() { return 5; }");
    sccache_command()
        .args(&[
            std::env::var("CC").unwrap_or("gcc".to_string()).as_str(),
            "-c",
            "-DSCCACHE_TEST_DEFINE",
        ])
        .arg(tmpdir.join(source_file))
        .arg("-o")
        .arg(tmpdir.join(obj_file))
        .envs(envs)
        .assert()
        .success();
}

pub fn dist_test_sccache_client_cfg(
    tmpdir: &Path,
    scheduler_url: HTTPUrl,
) -> sccache::config::FileConfig {
    let mut sccache_cfg = harness::sccache_client_cfg(tmpdir);
    sccache_cfg.cache.disk.as_mut().unwrap().size = 0;
    sccache_cfg.dist.scheduler_url = Some(scheduler_url);
    sccache_cfg
}

#[test]
#[cfg_attr(not(feature = "dist-tests"), ignore)]
fn test_dist_basic() {
    let tmpdir = tempfile::Builder::new()
        .prefix("sccache_dist_test")
        .tempdir()
        .unwrap();
    let tmpdir = tmpdir.path();
    let sccache_dist = harness::sccache_dist_path();

    let mut system = harness::DistSystem::new(&sccache_dist, tmpdir);
    system.add_scheduler();
    system.add_server();

    let sccache_cfg = dist_test_sccache_client_cfg(tmpdir, system.scheduler_url());
    let sccache_cfg_path = tmpdir.join("sccache-cfg.json");
    write_json_cfg(tmpdir, "sccache-cfg.json", &sccache_cfg);
    let sccache_cached_cfg_path = tmpdir.join("sccache-cached-cfg");

    stop_local_daemon();
    start_local_daemon(&sccache_cfg_path, &sccache_cached_cfg_path);
    basic_compile(tmpdir, &sccache_cfg_path, &sccache_cached_cfg_path);

    get_stats(|info| {
        assert_eq!(1, info.stats.dist_compiles.values().sum::<usize>());
        assert_eq!(0, info.stats.dist_errors);
        assert_eq!(1, info.stats.compile_requests);
        assert_eq!(1, info.stats.requests_executed);
        assert_eq!(0, info.stats.cache_hits.all());
        assert_eq!(1, info.stats.cache_misses.all());
    });
}

#[test]
#[cfg_attr(not(feature = "dist-tests"), ignore)]
fn test_dist_restartedserver() {
    let tmpdir = tempfile::Builder::new()
        .prefix("sccache_dist_test")
        .tempdir()
        .unwrap();
    let tmpdir = tmpdir.path();
    let sccache_dist = harness::sccache_dist_path();

    let mut system = harness::DistSystem::new(&sccache_dist, tmpdir);
    system.add_scheduler();
    let server_handle = system.add_server();

    let sccache_cfg = dist_test_sccache_client_cfg(tmpdir, system.scheduler_url());
    let sccache_cfg_path = tmpdir.join("sccache-cfg.json");
    write_json_cfg(tmpdir, "sccache-cfg.json", &sccache_cfg);
    let sccache_cached_cfg_path = tmpdir.join("sccache-cached-cfg");

    stop_local_daemon();
    start_local_daemon(&sccache_cfg_path, &sccache_cached_cfg_path);
    basic_compile(tmpdir, &sccache_cfg_path, &sccache_cached_cfg_path);

    system.restart_server(&server_handle);
    basic_compile(tmpdir, &sccache_cfg_path, &sccache_cached_cfg_path);

    get_stats(|info| {
        assert_eq!(2, info.stats.dist_compiles.values().sum::<usize>());
        assert_eq!(0, info.stats.dist_errors);
        assert_eq!(2, info.stats.compile_requests);
        assert_eq!(2, info.stats.requests_executed);
        assert_eq!(0, info.stats.cache_hits.all());
        assert_eq!(2, info.stats.cache_misses.all());
    });
}

#[test]
#[cfg_attr(not(feature = "dist-tests"), ignore)]
fn test_dist_nobuilder() {
    let tmpdir = tempfile::Builder::new()
        .prefix("sccache_dist_test")
        .tempdir()
        .unwrap();
    let tmpdir = tmpdir.path();
    let sccache_dist = harness::sccache_dist_path();

    let mut system = harness::DistSystem::new(&sccache_dist, tmpdir);
    system.add_scheduler();

    let sccache_cfg = dist_test_sccache_client_cfg(tmpdir, system.scheduler_url());
    let sccache_cfg_path = tmpdir.join("sccache-cfg.json");
    write_json_cfg(tmpdir, "sccache-cfg.json", &sccache_cfg);
    let sccache_cached_cfg_path = tmpdir.join("sccache-cached-cfg");

    stop_local_daemon();
    start_local_daemon(&sccache_cfg_path, &sccache_cached_cfg_path);
    basic_compile(tmpdir, &sccache_cfg_path, &sccache_cached_cfg_path);

    get_stats(|info| {
        assert_eq!(0, info.stats.dist_compiles.values().sum::<usize>());
        assert_eq!(1, info.stats.dist_errors);
        assert_eq!(1, info.stats.compile_requests);
        assert_eq!(1, info.stats.requests_executed);
        assert_eq!(0, info.stats.cache_hits.all());
        assert_eq!(1, info.stats.cache_misses.all());
    });
}

struct FailingServer;
impl ServerIncoming for FailingServer {
    fn handle_assign_job(&self, _job_id: JobId, _tc: Toolchain) -> Result<AssignJobResult> {
        let need_toolchain = false;
        let state = JobState::Ready;
        Ok(AssignJobResult {
            need_toolchain,
            state,
        })
    }
    fn handle_submit_toolchain(
        &self,
        _requester: &dyn ServerOutgoing,
        _job_id: JobId,
        _tc_rdr: ToolchainReader,
    ) -> Result<SubmitToolchainResult> {
        panic!("should not have submitted toolchain")
    }
    fn handle_run_job(
        &self,
        requester: &dyn ServerOutgoing,
        job_id: JobId,
        _command: CompileCommand,
        _outputs: Vec<String>,
        _inputs_rdr: InputsReader,
    ) -> Result<RunJobResult> {
        requester
            .do_update_job_state(job_id, JobState::Started)
            .context("Updating job state failed")?;
        bail!("internal build failure")
    }
}

#[test]
#[cfg_attr(not(feature = "dist-tests"), ignore)]
fn test_dist_failingserver() {
    let tmpdir = tempfile::Builder::new()
        .prefix("sccache_dist_test")
        .tempdir()
        .unwrap();
    let tmpdir = tmpdir.path();
    let sccache_dist = harness::sccache_dist_path();

    let mut system = harness::DistSystem::new(&sccache_dist, tmpdir);
    system.add_scheduler();
    system.add_custom_server(FailingServer);

    let sccache_cfg = dist_test_sccache_client_cfg(tmpdir, system.scheduler_url());
    let sccache_cfg_path = tmpdir.join("sccache-cfg.json");
    write_json_cfg(tmpdir, "sccache-cfg.json", &sccache_cfg);
    let sccache_cached_cfg_path = tmpdir.join("sccache-cached-cfg");

    stop_local_daemon();
    start_local_daemon(&sccache_cfg_path, &sccache_cached_cfg_path);
    basic_compile(tmpdir, &sccache_cfg_path, &sccache_cached_cfg_path);

    get_stats(|info| {
        assert_eq!(0, info.stats.dist_compiles.values().sum::<usize>());
        assert_eq!(1, info.stats.dist_errors);
        assert_eq!(1, info.stats.compile_requests);
        assert_eq!(1, info.stats.requests_executed);
        assert_eq!(0, info.stats.cache_hits.all());
        assert_eq!(1, info.stats.cache_misses.all());
    });
}
// -*- mode: rust; -*-
//
// This file is part of ocelot.
// Copyright © 2019 Galois, Inc.
// See LICENSE for licensing information.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "nightly", feature(stdsimd))]
#![cfg_attr(feature = "nightly", feature(external_doc))]
#![cfg_attr(feature = "nightly", doc(include = "../README.md"))]
#![cfg_attr(feature = "nightly", deny(missing_docs))]

//!

mod errors;
mod utils;

pub use crate::errors::Error;
pub mod oprf;
pub mod ot;
use std::{
    error::Error,
    fmt::Debug,
    intrinsics::write_bytes,
    sync::{Arc, Mutex, mpsc::{channel, Receiver, sync_channel}},
    thread::{self, yield_now},
};

use futures_util::{
    stream::{SplitSink, SplitStream},
    Sink, SinkExt, Stream, StreamExt, TryFutureExt, TryStreamExt,
};
use scuttlebutt::AbstractChannel;
use tokio::sync::mpsc::{ UnboundedSender, self};

#[derive(Debug)]
pub struct WSMessage {
    inner: Vec<u8>,
}

impl From<Vec<u8>> for WSMessage {
    fn from(v: Vec<u8>) -> Self {
        Self { inner: v }
    }
}

impl From<tokio_tungstenite::tungstenite::Message> for WSMessage {
    fn from(m: tokio_tungstenite::tungstenite::Message) -> Self {
        Self {
            inner: m.into_data(),
        }
    }
}

impl From<warp::ws::Message> for WSMessage {
    fn from(m: warp::ws::Message) -> Self {
        Self {
            inner: m.into_bytes(),
        }
    }
}

impl From<WSMessage> for warp::ws::Message {
    fn from(m: WSMessage) -> Self {
        warp::ws::Message::binary(m.inner)
    }
}

impl From<WSMessage> for tokio_tungstenite::tungstenite::Message {
    fn from(m: WSMessage) -> Self {
        tokio_tungstenite::tungstenite::Message::Binary(m.inner)
    }
}

#[derive(Debug, Clone)]
pub struct WSChannel {
    rx: Arc<Mutex<Receiver<WSMessage>>>,
    tx: UnboundedSender<WSMessage>,
    flush: Arc<Mutex<u64>>,
    sent:u64,
}

impl WSChannel {
    pub async fn from_split<T, W, E>(
        (mut user_ws_tx, mut user_ws_rx): (SplitSink<W, T>, SplitStream<W>),
    ) -> Self
    where
        T: From<WSMessage> + Send + Sync + 'static,
        W: Sink<T> + Sized + Unpin + Stream + Send + 'static,
        <W as futures_util::Sink<T>>::Error: std::fmt::Display,
        E: Error + Send,
        W: futures_util::Stream<Item = Result<T, E>>,
        WSMessage: From<T>,
    {
        let flush = Arc::new(Mutex::new(0));
        let (write_tx, mut rx) = mpsc::unbounded_channel::<WSMessage>();
        let (tx, mut read_rx) = sync_channel(0);
        let f = flush.clone();
        tokio::task::spawn(async move {
            loop {
                match rx.try_recv() {
                    
                    Ok(msg) => {
                        *f.lock().unwrap() +=1;
                        //println!("sent: {:?}",&msg);
                        user_ws_tx
                            .send(msg.into())
                            .unwrap_or_else(|e| {
                                eprintln!("websocket send error: {}", e);
                            })
                            .await;
                    }
                    Err(e) => match e {
                        mpsc::error::TryRecvError::Empty => {
                            continue;
                        }
                        mpsc::error::TryRecvError::Disconnected => break,
                    },
                }
            }
            println!("ERROR");
        });

        let r_handle = tokio::task::spawn(async move {
            while let Some(message) = user_ws_rx.next().await{
                let msg = match message {
                    Ok(msg) => msg,
                    Err(e) => {
                        println!("e:{}", e);
                        break;
                    }
                };

                let msg: WSMessage = msg.into();
                //println!("Recv: {:?}",msg.inner);
                tx.send(msg).unwrap();
            }
            println!("ERROR RECV");
        });
        Self {
            rx: Arc::new(Mutex::new(read_rx)),
            tx: write_tx,
            flush,
            sent:0
                
        }
    }
}

impl AbstractChannel for WSChannel {
    fn read_bytes(&mut self, bytes: &mut [u8]) -> std::io::Result<()> {
        println!("read");

        self.flush().unwrap();
        let message = self.rx.lock().unwrap().recv();

        if let Ok(mut message) = message {

            println!("Message {:?}", message.inner);
            message.inner.resize(bytes.len(), 0);
            bytes.copy_from_slice(&message.inner);
            println!("bytes: {:?}", bytes);
        }
        self.flush().unwrap();
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        println!("Write {:?}", bytes);
        self.flush();
        self.tx.send(bytes.to_vec().into()).unwrap();
        self.sent+=1;
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        println!("flush");
        while self.sent > *self.flush.lock().unwrap(){
            yield_now();
        }
        Ok(())
    }

    fn clone(&self) -> Self
    where
        Self: Sized,
    {
        println!("clone");
        Clone::clone(&self)
    }
}
use std::sync::Arc;

use warp::Filter;

use lazy_static::lazy_static;

use tera::Tera;

mod util;

mod handlers;
mod routes;
mod models;

lazy_static! {
    pub(crate) static ref TEMPLATES: Arc<Tera> = {
        let mut tera = match Tera::new("templates/**/*") {
            Ok(t) => t,
            Err(e) => {
                println!("Parsing error(s): {}", e);
                ::std::process::exit(1);
            }
        };
        if cfg!(feature = "autoreload"){
            tera.full_reload().unwrap();
        }
        Arc::new(tera)
    };
}

#[tokio::main]
async fn main() {

    util::tracing::start_tracing();
    let routes = routes::routes().with(warp::trace::request());
    util::starter::start_server(routes).await;
}

use keys_grpc_rs::KeysManager;
use async_trait::async_trait;
use blake3::Hash;
use fuse::FileType;
use futures::future::join_all;
use std::{
    error::Error,
    fmt,
    fs::{self, File},
    io::{self, prelude::*},
    path::Path,
    time::SystemTime,
};
use walkdir::WalkDir;

#[derive(Debug)]
pub struct DataSourceError {
    msg: String,
}

impl DataSourceError {
    pub fn new(msg: &str) -> Self {
        Self {
            msg: msg.to_string(),
        }
    }
}

impl Error for DataSourceError {}

impl fmt::Display for DataSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "File error {}!", self.msg)
    }
}

impl From<io::Error> for DataSourceError {
    fn from(error: io::Error) -> Self {
        DataSourceError {
            msg: error.to_string(),
        }
    }
}

pub fn hash_from_string(string: String) -> Hash {
    let mut arr: [u8; 32] = [0; 32];
    let bytes = &hex::decode(string).unwrap()[0..32];
    arr.copy_from_slice(bytes);
    Hash::from(arr)
}

#[derive(Debug)]
pub enum Area {
    Transient,
    Stage,
}

#[async_trait]
pub trait DataSource {
    fn get_area_path(&self, area: &Area) -> &String;
    fn get_keys_manager(&self) -> &KeysManager;
    fn get_data_from_source(&self, hash: &Hash) -> Result<Vec<u8>, DataSourceError>;
    async fn put_data_from_source(
        &self,
        hash: Hash,
        data: Vec<u8>,
    ) -> Result<(), DataSourceError>;
    async fn delete_data_from_source(&self, hash: &Hash) -> Result<(), DataSourceError>;
    fn get_data_attr_from_source(
        &self,
        hash: &Hash,
    ) -> Result<(u64, SystemTime), DataSourceError>;
    fn put_log_source(&self, log: Vec<u8>) -> Result<(), DataSourceError>;
    fn get_log_source(&self) -> Result<Vec<u8>, DataSourceError>;

    fn get_manager_user(&self) -> Option<String> {
        Some(self.get_keys_manager().get_user()?)
    }

    fn put_log(&self, log: String) {
        let log = self
            .get_keys_manager()
            .encrypt(log.as_bytes().to_vec(), None, false);
        let absolute_path = format!("{}/.log", self.get_area_path(&Area::Stage));
        let mut file = File::create(&absolute_path).unwrap();
        match file.write_all(&log) {
            Ok(()) => {}
            Err(e) => {
                println!("Error writing to log {}", e);
            }
        };
    }

    fn get_log(&self) -> Vec<(Hash, String, FileType)> {
        let absolute_path = format!("{}/.log", self.get_area_path(&Area::Stage));
        let entries_enc;
        if Path::new(&absolute_path).exists() {
            entries_enc = fs::read(&absolute_path).unwrap();
        } else {
            entries_enc = match self.get_log_source() {
                Ok(log) => log,
                _ => return vec![],
            };
        }
        let entries_enc = self.get_keys_manager().decrypt(entries_enc, false);
        let entries_string = std::str::from_utf8(&entries_enc).unwrap();
        let entries: Vec<&str> = entries_string.split("\n").collect();
        let mut nodes: Vec<(Hash, String, FileType)> = vec![];
        for entry in entries {
            let entry: Vec<&str> = entry.split("\t").collect();
            if entry.len() == 3 {
                let kind = match entry[0] {
                    "file" => FileType::RegularFile,
                    "dir" => FileType::Directory,
                    _ => {
                        continue;
                    }
                };
                let hash = hash_from_string(entry[1].to_string());
                let key = entry[2][1..entry[2].len() - 1].to_string();
                nodes.push((hash, key, kind));
            }
        }
        nodes
    }

    fn check_in_area(&self, hash: &Hash, area: &Area) -> bool {
        let absolute_path = format!("{}/{}", self.get_area_path(area), hash.to_hex());
        Path::new(&absolute_path).exists()
    }

    fn get_data_from_area(
        &self,
        hash: &Hash,
        area: &Area,
    ) -> Result<Vec<u8>, DataSourceError> {
        let absolute_path = format!("{}/{}", self.get_area_path(area), hash.to_hex());
        let buf = fs::read(&absolute_path)?;
        Ok(buf)
    }

    fn put_data_area(&self, hash: &Hash, data: &[u8], area: &Area) {
        let absolute_path = format!("{}/{}", self.get_area_path(area), hash.to_hex());
        println!("Create in area: {}", absolute_path);
        let mut file = File::create(&absolute_path).unwrap();
        match file.write_all(data) {
            Ok(()) => {}
            Err(e) => {
                println!("Error writing to area {:?}, {}", area, e);
            }
        };
    }

    fn put_hash_data_area(&self, data: &Vec<u8>, area: &Area) -> Hash {
        let hash = blake3::hash(&data);
        self.put_data_area(&hash, &data, area);
        hash
    }

    fn delete_data_area(&self, hash: &Hash, area: &Area) -> Result<(), DataSourceError> {
        let absolute_path = format!("{}/{}", self.get_area_path(area), hash.to_hex());
        if Path::new(&absolute_path).exists() {
            fs::remove_file(&absolute_path)?;
        }
        Ok(())
    }

    fn get_hashes_area(&self, area: &Area) -> Result<Vec<Hash>, DataSourceError> {
        let absolute_path = format!("{}/", self.get_area_path(area));
        let dir = WalkDir::new(&absolute_path);
        let mut hashes = vec![];
        for e in dir.into_iter().filter_map(|e| e.ok()) {
            if e.metadata().unwrap().is_file() {
                if e.file_name() == ".log" {
                    continue;
                }
                let string = e
                    .path()
                    .to_str()
                    .unwrap()
                    .replace(&absolute_path, "")
                    .to_string();
                hashes.push(hash_from_string(string));
            }
        }
        Ok(hashes)
    }
    fn get_data_attr_area(
        &self,
        hash: &Hash,
        area: &Area,
    ) -> Result<(u64, SystemTime), DataSourceError> {
        let absolute_path = format!("{}/{}", self.get_area_path(area), hash.to_hex());
        let file = fs::metadata(&absolute_path)?;
        Ok((file.len(), file.modified()?))
    }

    fn get_data(&self, hash: &Hash) -> Result<Vec<u8>, DataSourceError> {
        println!("Get data: {}", hash.to_hex());
        let data;
        if self.check_in_area(&hash, &Area::Stage) {
            data = self.get_data_from_area(&hash, &Area::Stage).unwrap();
        } else if self.check_in_area(&hash, &Area::Transient) {
            data = self.get_data_from_area(&hash, &Area::Transient).unwrap();
        } else {
            let block_data = self.get_data_from_source(&hash)?;
            self.put_data_area(&hash, &block_data, &Area::Transient);
            data = block_data;
        }
        let data = self.get_keys_manager().decrypt(data, false);
        Ok(data)
    }
    fn put_data(&self, data: Vec<u8>, recipients:Option<Vec<String>>) -> Hash {
        let data = self.get_keys_manager().encrypt(data, recipients, false);
        self.put_hash_data_area(&data, &Area::Stage)
    }

    async fn delete_data(&self, hash: &Hash) -> Result<(), DataSourceError> {
        self.delete_data_area(hash, &Area::Transient)?;
        self.delete_data_area(hash, &Area::Stage)?;
        self.delete_data_from_source(hash).await?;
        Ok(())
    }

    fn get_data_attr(&self, hash: &Hash) -> Result<(u64, SystemTime), DataSourceError> {
        if self.check_in_area(&hash, &Area::Stage) {
            Ok(self.get_data_attr_area(hash, &Area::Stage)?)
        } else {
            Ok(self.get_data_attr_from_source(hash)?)
        }
    }

    async fn sync_stage_area(
        &self,
        valid_hashes: Vec<Hash>,
    ) -> Result<(), DataSourceError> {
        let mut futs = vec![];
        let hashes = self.get_hashes_area(&Area::Stage)?;
        for hash in &hashes {
            if !valid_hashes.contains(hash) {
                self.delete_data_area(hash, &Area::Stage)?;
                continue;
            }
            let data = self.get_data_from_area(&hash, &Area::Stage)?;
            println!("Syncing: {}", hash.to_hex());
            let hash = hash.clone();
            let fut = self.put_data_from_source(hash, data);
            futs.push(fut);
        }
        let log_path = format!("{}/.log", self.get_area_path(&Area::Stage));
        if Path::new(&log_path).exists() {
            let data = fs::read(&log_path).unwrap();
            self.put_log_source(data)?;
            fs::remove_file(&log_path)?;
        }
        for res in join_all(futs).await {
            res?;
        }
        for hash in hashes {
            self.delete_data_area(&hash, &Area::Stage)?;
        }
        Ok::<(), DataSourceError>(())
    }
}

pub mod sources {
    pub mod s3_bucket {
        use crate::datasource::{Area, DataSource, DataSourceError};
        use async_trait::async_trait;
        use s3::{bucket::Bucket, S3Error};
        use std::time::{Duration, UNIX_EPOCH};
        use keys_grpc_rs::KeysManager;
        use blake3::Hash;

        impl From<S3Error> for DataSourceError {
            fn from(err: S3Error) -> Self {
                Self {
                    msg: err.to_string(),
                }
            }
        }

        pub struct BucketSource {
            bucket: Bucket,
            transient_path: String,
            stage_path: String,
            keys_manager: KeysManager,
        }

        impl BucketSource {
            pub fn new(
                bucket: Bucket,
                transient_path: String,
                stage_path: String,
                keys_manager: KeysManager,
            ) -> Self {
                Self {
                    bucket,
                    transient_path,
                    stage_path,
                    keys_manager,
                }
            }
        }

        #[async_trait]
        impl DataSource for BucketSource {
            fn get_area_path(&self, area: &Area) -> &String {
                match area {
                    Area::Transient => &self.transient_path,
                    Area::Stage => &self.stage_path,
                }
            }
            fn get_data_from_source(
                &self,
                hash: &Hash,
            ) -> Result<Vec<u8>, DataSourceError> {
                let (result, code) = self.bucket.get_object_blocking(hash.to_hex())?;
                if code != 200 {
                    Err(DataSourceError::new(&format!("Error wrong code: {}", code)))
                } else {
                    Ok(result)
                }
            }
            async fn put_data_from_source(
                &self,
                hash: Hash,
                data: Vec<u8>,
            ) -> Result<(), DataSourceError> {
                let (_, code) =
                    self.bucket
                        .put_object_blocking(hash.to_hex(), &data, "text/plain")?;
                if code != 200 {
                    Err(DataSourceError::new(&format!("Error wrong code: {}", code)))
                } else {
                    Ok(())
                }
            }

            async fn delete_data_from_source(
                &self,
                hash: &Hash,
            ) -> Result<(), DataSourceError> {
                let (_, code) = self.bucket.delete_object_blocking(hash.to_hex())?;
                if code != 204 {
                    Err(DataSourceError::new(&format!("Error wrong code: {}", code)))
                } else {
                    Ok(())
                }
            }

            fn get_data_attr_from_source(
                &self,
                hash: &Hash,
            ) -> Result<(u64, std::time::SystemTime), DataSourceError> {
                let results = self
                    .bucket
                    .list_blocking(format!("{}", hash.to_hex()), None)?;
                let (result, code) = results.first().unwrap();
                if code != &200 {
                    return Err(DataSourceError::new(&format!("Error wrong code: {}", code)));
                }
                let obj = match result.contents.first() {
                    Some(obj) => obj,
                    None => return Err(DataSourceError::new("S3 list blocking empty")),
                };
                let time = chrono::DateTime::parse_from_rfc3339(&obj.last_modified).unwrap();
                let time = UNIX_EPOCH + Duration::from_millis(time.timestamp_millis() as u64);
                let size = obj.size;
                Ok((size, time))
            }

            fn put_log_source(&self, log: Vec<u8>) -> Result<(), DataSourceError> {
                let (_, code) = self
                    .bucket
                    .put_object_blocking(".log", &log, "text/plain")?;
                if code != 200 {
                    Err(DataSourceError::new(&format!("Error wrong code: {}", code)))
                } else {
                    Ok(())
                }
            }
            fn get_log_source(&self) -> Result<Vec<u8>, DataSourceError> {
                let (result, code) = self.bucket.get_object_blocking(".log")?;
                if code != 200 {
                    Err(DataSourceError::new(&format!("Error wrong code: {}", code)))
                } else {
                    Ok(result)
                }
            }
            fn get_keys_manager(&self) -> &KeysManager {
                &self.keys_manager
            }
        }
    }
}
fn main() {
}

fn filter((x,y):(u32),)
#![no_std]
#![no_main]

extern crate panic_halt;

use riscv_rt::entry;
use hifive1::hal::prelude::*;
use hifive1::hal::DeviceResources;

#[entry]
fn main() -> ! {
    let dr = DeviceResources::take().unwrap();
    let p = dr.peripherals;

    // Configure clocks
    let _clocks = hifive1::clock::configure(p.PRCI, p.AONCLK, 320.mhz().into());

    loop {}
}
use futures_util::StreamExt;
use mpc_cache::WSChannel;
use popsicle::psty::{Sender, Receiver};
use scuttlebutt::AesRng;
use tokio::task::yield_now;
use tokio_tungstenite::connect_async;

#[tokio::main]
async fn main() {
    let (ws,_) = connect_async("wss://127.0.0.1:3030/psty").await.expect("Failed to connect");
    let mut rng = AesRng::new();
    let mut channel = WSChannel::from_split(ws.split()).await;
    yield_now().await;
    let mut sender = Receiver::init(&mut channel, &mut rng).unwrap();
    yield_now().await;
    println!("-------------------------------------------------------------------");
    let inputs = vec![vec![0,1,2],vec![3,4,5]];
    let state = sender.receive(&inputs,&mut channel, &mut rng).unwrap();
    yield_now().await;
    println!("-------------------------------------------------------------------");
    let result = state.compute_intersection(&mut channel, &mut rng).unwrap();
    println!("Result {:?}",result);
}
// -*- mode: rust; -*-
//
// This file is part of `popsicle`.
// Copyright © 2019 Galois, Inc.
// See LICENSE for licensing information.

#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "nightly", feature(external_doc))]
#![cfg_attr(feature = "nightly", doc(include = "../README.md"))]
#![cfg_attr(feature = "nightly", deny(missing_docs))]

//!

mod cuckoo;
mod errors;
mod psi;
pub mod utils;

pub use crate::{errors::Error, psi::*};
// -*- mode: rust; -*-
//
// This file is part of ocelot.
// Copyright © 2019 Galois, Inc.
// See LICENSE for licensing information.

use scuttlebutt::Block;

#[inline]
pub fn transpose(m: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let mut m_ = vec![0u8; nrows * ncols / 8];
    _transpose(
        m_.as_mut_ptr() as *mut u8,
        m.as_ptr(),
        nrows as u64,
        ncols as u64,
    );
    m_
}

#[inline(always)]
fn _transpose(out: *mut u8, inp: *const u8, nrows: u64, ncols: u64) {
    assert!(nrows >= 16);
    assert_eq!(nrows % 8, 0);
    assert_eq!(ncols % 8, 0);
    unsafe { sse_trans(out, inp, nrows, ncols) }
}

#[link(name = "transpose")]
extern "C" {
    fn sse_trans(out: *mut u8, inp: *const u8, nrows: u64, ncols: u64);
}

// The hypothesis that a rust implementation of matrix transpose would be faster
// than the C implementation appears to be false... But let's leave this code
// here for now just in case.

// union __U128 {
//     vector: __m128i,
//     bytes: [u8; 16],
// }

// impl Default for __U128 {
//     #[inline]
//     fn default() -> Self {
//         __U128 { bytes: [0u8; 16] }
//     }
// }

// #[inline]
// pub fn transpose(input: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
//     assert_eq!(nrows % 16, 0);
//     assert_eq!(ncols % 16, 0);
//     let mut output = vec![0u8; nrows * ncols / 8];
//     unsafe {
//         let mut h: &[u8; 4];
//         let mut v: __m128i;
//         let mut rr: usize = 0;
//         let mut cc: usize;
//         while rr <= nrows - 16 {
//             cc = 0;
//             while cc < ncols {
//                 v = _mm_set_epi8(
//                     input[(rr + 15) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 14) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 13) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 12) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 11) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 10) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 9) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 8) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 7) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 6) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 5) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 4) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 3) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 2) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 1) * ncols / 8 + cc / 8] as i8,
//                     input[(rr + 0) * ncols / 8 + cc / 8] as i8,
//                 );
//                 for i in (0..8).rev() {
//                     h = &*(&_mm_movemask_epi8(v) as *const _ as *const [u8; 4]);
//                     output[(cc + i) * nrows / 8 + rr / 8] = h[0];
//                     output[(cc + i) * nrows / 8 + rr / 8 + 1] = h[1];
//                     v = _mm_slli_epi64(v, 1);
//                 }
//                 cc += 8;
//             }
//             rr += 16;
//         }
//         if rr == nrows {
//             return output;
//         }

//         cc = 0;
//         while cc <= ncols - 16 {
//             let mut v = _mm_set_epi16(
//                 input[((rr + 7) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 6) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 5) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 4) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 3) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 2) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 1) * ncols / 8 + cc / 8) / 2] as i16,
//                 input[((rr + 0) * ncols / 8 + cc / 8) / 2] as i16,
//             );
//             for i in (0..8).rev() {
//                 h = &*(&_mm_movemask_epi8(v) as *const _ as *const [u8; 4]);
//                 output[(cc + i) * nrows / 8 + rr / 8] = h[0];
//                 output[(cc + i) * nrows / 8 + rr / 8 + 8] = h[1];
//                 v = _mm_slli_epi64(v, 1);
//             }
//             cc += 16;
//         }
//         if cc == ncols {
//             return output;
//         }
//         let mut tmp = __U128 {
//             bytes: [
//                 input[(rr + 0) * ncols / 8 + cc / 8],
//                 input[(rr + 1) * ncols / 8 + cc / 8],
//                 input[(rr + 2) * ncols / 8 + cc / 8],
//                 input[(rr + 3) * ncols / 8 + cc / 8],
//                 input[(rr + 4) * ncols / 8 + cc / 8],
//                 input[(rr + 5) * ncols / 8 + cc / 8],
//                 input[(rr + 6) * ncols / 8 + cc / 8],
//                 input[(rr + 7) * ncols / 8 + cc / 8],
//                 0u8,
//                 0u8,
//                 0u8,
//                 0u8,
//                 0u8,
//                 0u8,
//                 0u8,
//                 0u8,
//             ],
//         };
//         for i in (0..8).rev() {
//             h = &*(&_mm_movemask_epi8(tmp.vector) as *const _ as *const [u8; 4]);
//             output[(cc + i) * nrows / 8 + rr / 8] = h[0];
//             tmp.vector = _mm_slli_epi64(tmp.vector, 1);
//         }
//     };
//     output
// }

#[inline]
pub fn boolvec_to_u8vec(bv: &[bool]) -> Vec<u8> {
    let offset = if bv.len() % 8 == 0 { 0 } else { 1 };
    let mut v = vec![0u8; bv.len() / 8 + offset];
    for (i, b) in bv.iter().enumerate() {
        v[i / 8] |= (*b as u8) << (i % 8);
    }
    v
}
#[inline]
pub fn u8vec_to_boolvec(v: &[u8]) -> Vec<bool> {
    let mut bv = Vec::with_capacity(v.len() * 8);
    for byte in v.iter() {
        for i in 0..8 {
            bv.push((1 << i) & byte != 0);
        }
    }
    bv
}

#[inline(always)]
pub fn xor_two_blocks(x: &(Block, Block), y: &(Block, Block)) -> (Block, Block) {
    (x.0 ^ y.0, x.1 ^ y.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _transpose(nrows: usize, ncols: usize) {
        let m = (0..nrows * ncols / 8)
            .map(|_| rand::random::<u8>())
            .collect::<Vec<u8>>();
        let m_ = m.clone();
        let m = transpose(&m, nrows, ncols);
        let m = transpose(&m, ncols, nrows);
        assert_eq!(m, m_);
    }

    #[test]
    fn test_transpose() {
        _transpose(16, 16);
        _transpose(24, 16);
        _transpose(32, 16);
        _transpose(40, 16);
        _transpose(128, 16);
        _transpose(128, 24);
        _transpose(128, 128);
        _transpose(128, 1 << 16);
        _transpose(128, 1 << 18);
        _transpose(32, 32);
        _transpose(64, 32);
    }

    #[test]
    fn test_boolvec_to_u8vec() {
        let v = (0..128)
            .map(|_| rand::random::<bool>())
            .collect::<Vec<bool>>();
        let v_ = boolvec_to_u8vec(&v);
        let v__ = u8vec_to_boolvec(&v_);
        assert_eq!(v, v__);
    }

    #[test]
    fn test_u8vec_to_boolvec() {
        let v = (0..128).map(|_| rand::random::<u8>()).collect::<Vec<u8>>();
        let v_ = u8vec_to_boolvec(&v);
        let v__ = boolvec_to_u8vec(&v_);
        assert_eq!(v, v__);
    }
}

#[cfg(all(feature = "nightly", test))]
mod benchmarks {
    extern crate test;
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_transpose(b: &mut Bencher) {
        let (nrows, ncols) = (128, 1 << 18);
        let m = (0..nrows * ncols / 8)
            .map(|_| rand::random::<u8>())
            .collect::<Vec<u8>>();
        b.iter(|| transpose(&m, nrows, ncols));
    }
}
mod routes;
mod handlers;


#[tokio::main]
async fn main()  {
    let routes = routes::routes().await;
    // The warp_lambda::run() function takes care of invoking the aws lambda runtime for you
    let warp_service = warp::service(routes);
    warp_lambda::run(warp_service)
        .await
        .expect("An error occured");
}
use std::sync::Arc;

use crate::error::ImageDataError;
use crate::ImageInfo;

/// Trait for borrowing image data from a struct.
pub trait AsImageView {
	/// Get an image view for the object.
	fn as_image_view(&self) -> Result<ImageView, ImageDataError>;
}

/// Get the image info of an object that implements [`AsImageView`].
pub fn image_info(image: &impl AsImageView) -> Result<ImageInfo, ImageDataError> {
	Ok(image.as_image_view()?.info())
}

/// Borrowed view of image data,
#[derive(Debug, Copy, Clone)]
pub struct ImageView<'a> {
	info: ImageInfo,
	data: &'a [u8],
}

impl<'a> ImageView<'a> {
	/// Create a new image view from image information and a data slice.
	pub fn new(info: ImageInfo, data: &'a [u8]) -> Self {
		Self { info, data }
	}

	/// Get the image information.
	pub fn info(&self) -> ImageInfo {
		self.info
	}

	/// Get the image data as byte slice.
	pub fn data(&self) -> &[u8] {
		self.data
	}
}

impl<'a> AsImageView for ImageView<'a> {
	fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		Ok(*self)
	}
}

/// Owning image that can be sent to another thread.
///
/// The image is backed by either a [`Box`] or [`Arc`].
/// It can either directly own the data or through a [`dyn AsImageView`].
pub enum Image {
	/// An image backed by a `Box<[u8]>`.
	Box(BoxImage),

	/// An image backed by an `Arc<[u8]>`.
	Arc(ArcImage),

	/// An image backed by a `Box<dyn AsImageView>`.
	BoxDyn(Box<dyn AsImageView + Send>),

	/// An image backed by an `Arc<dyn AsImageView>`.
	ArcDyn(Arc<dyn AsImageView + Sync + Send>),

	/// An invalid image that will always fail the conversion to [`ImageView`].
	Invalid(ImageDataError),
}

impl Clone for Image {
	fn clone(&self) -> Self {
		match self {
			Self::Box(x) => Self::Box(x.clone()),
			Self::Arc(x) => Self::Arc(x.clone()),
			// We can not clone Box<dyn AsImageView> directly, but we can clone the data or the error.
			Self::BoxDyn(x) => match x.as_image_view() {
				Ok(view) => Self::Box(BoxImage::new(view.info, view.data.into())),
				Err(error) => Self::Invalid(error),
			},
			Self::ArcDyn(x) => Self::ArcDyn(x.clone()),
			Self::Invalid(x) => Self::Invalid(x.clone()),
		}
	}
}

impl<T: AsImageView> AsImageView for Box<T> {
	fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		self.as_ref().as_image_view()
	}
}

impl<T: AsImageView> AsImageView for Arc<T> {
	fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		self.as_ref().as_image_view()
	}
}

/// Image backed by a `Box<[u8]>`.
#[derive(Debug, Clone)]
pub struct BoxImage {
	info: ImageInfo,
	data: Box<[u8]>,
}

/// Image backed by an `Arc<[u8]>`.
#[derive(Debug, Clone)]
pub struct ArcImage {
	info: ImageInfo,
	data: Arc<[u8]>,
}

impl Image {
	/// Get a non-owning view of the image data.
	pub fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		match self {
			Self::Box(x) => Ok(x.as_view()),
			Self::Arc(x) => Ok(x.as_view()),
			Self::BoxDyn(x) => x.as_image_view(),
			Self::ArcDyn(x) => x.as_image_view(),
			Self::Invalid(e) => Err(e.clone()),
		}
	}
}

impl AsImageView for Image {
	fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		self.as_image_view()
	}
}

impl BoxImage {
	/// Create a new image from image information and a boxed slice.
	pub fn new(info: ImageInfo, data: Box<[u8]>) -> Self {
		Self { info, data }
	}

	/// Get a non-owning view of the image data.
	pub fn as_view(&self) -> ImageView {
		ImageView::new(self.info, &self.data)
	}

	/// Get the image information.
	pub fn info(&self) -> ImageInfo {
		self.info
	}

	/// Get the image data as byte slice.
	pub fn data(&self) -> &[u8] {
		&self.data
	}
}

impl AsImageView for BoxImage {
	fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		Ok(self.as_view())
	}
}

impl ArcImage {
	/// Create a new image from image information and a Arc-wrapped slice.
	pub fn new(info: ImageInfo, data: Arc<[u8]>) -> Self {
		Self { info, data }
	}

	/// Get a non-owning view of the image data.
	pub fn as_view(&self) -> ImageView {
		ImageView::new(self.info, &self.data)
	}

	/// Get the image information.
	pub fn info(&self) -> ImageInfo {
		self.info
	}

	/// Get the image data as byte slice.
	pub fn data(&self) -> &[u8] {
		&self.data
	}
}

impl AsImageView for ArcImage {
	fn as_image_view(&self) -> Result<ImageView, ImageDataError> {
		Ok(self.as_view())
	}
}

impl From<ImageView<'_>> for BoxImage {
	fn from(other: ImageView) -> Self {
		Self {
			info: other.info,
			data: other.data.into(),
		}
	}
}

impl From<&'_ ImageView<'_>> for BoxImage {
	fn from(other: &ImageView) -> Self {
		Self {
			info: other.info,
			data: other.data.into(),
		}
	}
}

impl From<ImageView<'_>> for ArcImage {
	fn from(other: ImageView) -> Self {
		Self {
			info: other.info,
			data: other.data.into(),
		}
	}
}

impl From<&'_ ImageView<'_>> for ArcImage {
	fn from(other: &ImageView) -> Self {
		Self {
			info: other.info,
			data: other.data.into(),
		}
	}
}

impl From<ImageView<'_>> for Image {
	fn from(other: ImageView) -> Self {
		Self::Box(BoxImage::from(other))
	}
}

impl From<&'_ ImageView<'_>> for Image {
	fn from(other: &ImageView) -> Self {
		Self::Box(BoxImage::from(other))
	}
}

impl From<BoxImage> for ArcImage {
	fn from(other: BoxImage) -> Self {
		Self {
			info: other.info,
			data: other.data.into(),
		}
	}
}

impl From<BoxImage> for Image {
	fn from(other: BoxImage) -> Self {
		Self::Box(other)
	}
}

impl From<ArcImage> for Image {
	fn from(other: ArcImage) -> Self {
		Self::Arc(other)
	}
}

impl From<Box<dyn AsImageView + Send>> for Image {
	fn from(other: Box<dyn AsImageView + Send>) -> Self {
		Self::BoxDyn(other)
	}
}

impl From<Arc<dyn AsImageView + Sync + Send>> for Image {
	fn from(other: Arc<dyn AsImageView + Sync + Send>) -> Self {
		Self::ArcDyn(other)
	}
}

impl<T> From<Box<T>> for Image
where
	T: AsImageView + Send + 'static,
{
	fn from(other: Box<T>) -> Self {
		Self::BoxDyn(other)
	}
}

impl<T> From<Arc<T>> for Image
where
	T: AsImageView + Send + Sync + 'static,
{
	fn from(other: Arc<T>) -> Self {
		Self::ArcDyn(other)
	}
}
use crate::handlers;
use biscuit::{
    jwa::SignatureAlgorithm,
    jwk::{AlgorithmParameters, JWKSet},
    jws::{RegisteredHeader, Secret},
    ClaimsSet, RegisteredClaims, JWT,
};
use jsonwebtoken::Validation;
use rusoto_core::credential::{EnvironmentProvider, ProvideAwsCredentials};
use rusoto_core::{credential::AwsCredentials, Region};
use rusoto_s3::S3Client;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use warp::{Filter, Rejection, Reply};

pub async fn routes(
) -> impl Filter<Extract = (impl Reply,), Error = Rejection> + Clone + Send + Sync + 'static {
    let client = S3Client::new(Region::UsEast2);
    let credentials = EnvironmentProvider::default().credentials().await.unwrap();
    //POST
    upload_quiz(credentials.clone())
        .or(upload_answer(credentials.clone()))
        .or(set_current(client.clone()))
        //GET
        .or(get_quiz(client.clone()))
        .or(get_answers(client.clone()))
        .or(get_current_quiz(client.clone()))
}

pub fn get_current_quiz(
    client: S3Client,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone + Send + Sync + 'static {
    warp::path!("current")
        .and(warp::get())
        .and(with_client(client))
        .and_then(handlers::get_current_quiz)
}

pub fn get_quiz(
    client: S3Client,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone + Send + Sync + 'static {
    warp::path!("quiz" / String)
        .and(warp::get())
        .and(with_client(client))
        .and_then(handlers::get_quiz)
}

pub fn get_answers(
    client: S3Client,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone + Send + Sync + 'static {
    warp::path!("answers" / String)
        .and(warp::get())
        .and(with_client(client))
        .and_then(handlers::get_answers)
}

pub fn upload_quiz(
    credentials: AwsCredentials,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone + Send + Sync + 'static {
    warp::path!("upload-quiz" / String)
        .and(warp::post())
        .map(|file| format!("{}/{}.md", "answers", file))
        .and(with_credentials(credentials))
        .and_then(handlers::upload)
}

pub fn upload_answer(
    credentials: AwsCredentials,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone + Send + Sync + 'static {
    warp::path!("upload-quiz" / String)
        .and(warp::post())
        .map(|file| format!("{}/{}.md", "answers", file))
        .and(with_credentials(credentials))
        .and_then(handlers::upload)
}

pub fn set_current(
    client: S3Client,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone + Send + Sync + 'static {
    warp::path!("set-current" / String)
        .and(warp::post())
        .and(with_client(client))
        .and_then(handlers::set_current)
}
pub fn with_client(
    client: S3Client,
) -> impl Filter<Extract = (S3Client,), Error = Infallible> + Clone + 'static {
    warp::any().map(move || client.clone())
}

pub fn with_credentials(
    credentials: AwsCredentials,
) -> impl Filter<Extract = (AwsCredentials,), Error = Infallible> + Clone {
    warp::any().map(move || credentials.clone())
}

pub fn auth(token: String) -> impl Filter<Extract = (), Error = Rejection> + Clone {}

pub fn with_auth(token: String) {}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
struct PrivateClaims {}

fn fetch_jwks(uri: &str) -> Result<JWKSet<PrivateClaims>, std::io::Error> {
    let mut res = reqwest::get(uri)?;
    let val = res.json::<JWKSet>()?;
    return Ok(val);
}

pub fn validate_token(token: String) -> Result<bool, std::io::Error> {
    let authority = std::env::var("AUTHORITY").expect("AUTHORITY must be set");
    let jwks = fetch_jwks(&format!(
        "{}{}",
        authority.as_str(),
        ".well-known/jwks.json"
    ))
    .expect("failed to fetch jwks");

    let expected_claims = ClaimsSet::<PrivateClaims> {
        registered: RegisteredClaims {
            issuer: Some(authority),
            ..Default::default()
        }..Default::default(),
        private: PrivateClaims {},
    };
    let expected_jwt = JWT::new_decoded(
        From::from(RegisteredHeader {
            algorithm: SignatureAlgorithm::HS256,
            ..Default::default()
        }),
        expected_claims.clone(),
    );
    let kid = expected_jwt
        .unverified_header()
        .unwrap()
        .registered
        .key_id
        .unwrap();
    let jwk = jwks.find(kid).unwrap();

    // Now that we have the key, construct our RSA public key secret
    let secret = match jwk.algorithm {
        AlgorithmParameters::RSA(ref rsa_key) => Secret::Pkcs {
            n: rsa_key.n.clone(),
            e: rsa_key.e.clone(),
        },
        _ => return Err(),
    };

    // Not fully verify and extract the token with verification
    let token = token.into_decoded(&secret, SignatureAlgorithm::RS256);
    let user = token.header();
}
use std::{error::Error, fs::File, io::{self, BufRead, Read, Write}, path::Path};

use clap::{Arg,App};

mod token;

static mut HAD_ERROR:bool = false;

fn main() {
    let matches = App::new("rlox")
        .version("1.0")
        .arg(Arg::with_name("source")
             .value_name("FILE")
             ).get_matches();
    if let Some(input) = matches.value_of("FILE"){
        run_file(Path::new(input));
    }else{
        run_prompt();
    }
}

fn run_file(path:&Path){
    let mut file = match File::open(path){
        Err(e) => panic!("Couldn't read source file {}: {}",path.to_str().unwrap(),e),
        Ok(file) => file
    };
    let mut s = String::new();
    file.read_to_string(&mut s).unwrap();
    run(s);
    unsafe{
        if HAD_ERROR {panic!("Error")};
    }
}

fn run(src: String){
    for t in src.split_whitespace(){

    }
}

fn run_prompt(){
    loop {
        let mut input = String::new();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut input).unwrap();
        run(input);
        unsafe{HAD_ERROR = false}
    }
}

fn error(line:u64, e:impl Error){
    eprintln!("{}: {}",line,e);
}

fn report(line:u64,loc:String, e:impl Error){
    eprintln!("[line {}] Error {}: {}",line,loc,e);
}
/*!
This crate provides a library for parsing, compiling, and executing regular
expressions. Its syntax is similar to Perl-style regular expressions, but lacks
a few features like look around and backreferences. In exchange, all searches
execute in linear time with respect to the size of the regular expression and
search text.

This crate's documentation provides some simple examples, describes
[Unicode support](#unicode) and exhaustively lists the
[supported syntax](#syntax).

For more specific details on the API for regular expressions, please see the
documentation for the [`Regex`](struct.Regex.html) type.

# Usage

This crate is [on crates.io](https://crates.io/crates/regex) and can be
used by adding `regex` to your dependencies in your project's `Cargo.toml`.

```toml
[dependencies]
regex = "1"
```

If you're using Rust 2015, then you'll also need to add it to your crate root:

```rust
extern crate regex;
```

# Example: find a date

General use of regular expressions in this package involves compiling an
expression and then using it to search, split or replace text. For example,
to confirm that some text resembles a date:

```rust
use regex::Regex;
let re = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
assert!(re.is_match("2014-01-01"));
```

Notice the use of the `^` and `$` anchors. In this crate, every expression
is executed with an implicit `.*?` at the beginning and end, which allows
it to match anywhere in the text. Anchors can be used to ensure that the
full text matches an expression.

This example also demonstrates the utility of
[raw strings](https://doc.rust-lang.org/stable/reference/tokens.html#raw-string-literals)
in Rust, which
are just like regular strings except they are prefixed with an `r` and do
not process any escape sequences. For example, `"\\d"` is the same
expression as `r"\d"`.

# Example: Avoid compiling the same regex in a loop

It is an anti-pattern to compile the same regular expression in a loop
since compilation is typically expensive. (It takes anywhere from a few
microseconds to a few **milliseconds** depending on the size of the
regex.) Not only is compilation itself expensive, but this also prevents
optimizations that reuse allocations internally to the matching engines.

In Rust, it can sometimes be a pain to pass regular expressions around if
they're used from inside a helper function. Instead, we recommend using the
[`lazy_static`](https://crates.io/crates/lazy_static) crate to ensure that
regular expressions are compiled exactly once.

For example:

```rust
#[macro_use] extern crate lazy_static;
extern crate regex;

use regex::Regex;

fn some_helper_function(text: &str) -> bool {
    lazy_static! {
        static ref RE: Regex = Regex::new("...").unwrap();
    }
    RE.is_match(text)
}

fn main() {}
```

Specifically, in this example, the regex will be compiled when it is used for
the first time. On subsequent uses, it will reuse the previous compilation.

# Example: iterating over capture groups

This crate provides convenient iterators for matching an expression
repeatedly against a search string to find successive non-overlapping
matches. For example, to find all dates in a string and be able to access
them by their component pieces:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
let text = "2012-03-14, 2013-01-01 and 2014-07-05";
for cap in re.captures_iter(text) {
    println!("Month: {} Day: {} Year: {}", &cap[2], &cap[3], &cap[1]);
}
// Output:
// Month: 03 Day: 14 Year: 2012
// Month: 01 Day: 01 Year: 2013
// Month: 07 Day: 05 Year: 2014
# }
```

Notice that the year is in the capture group indexed at `1`. This is
because the *entire match* is stored in the capture group at index `0`.

# Example: replacement with named capture groups

Building on the previous example, perhaps we'd like to rearrange the date
formats. This can be done with text replacement. But to make the code
clearer, we can *name*  our capture groups and use those names as variables
in our replacement text:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})").unwrap();
let before = "2012-03-14, 2013-01-01 and 2014-07-05";
let after = re.replace_all(before, "$m/$d/$y");
assert_eq!(after, "03/14/2012, 01/01/2013 and 07/05/2014");
# }
```

The `replace` methods are actually polymorphic in the replacement, which
provides more flexibility than is seen here. (See the documentation for
`Regex::replace` for more details.)

Note that if your regex gets complicated, you can use the `x` flag to
enable insignificant whitespace mode, which also lets you write comments:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"(?x)
  (?P<y>\d{4}) # the year
  -
  (?P<m>\d{2}) # the month
  -
  (?P<d>\d{2}) # the day
").unwrap();
let before = "2012-03-14, 2013-01-01 and 2014-07-05";
let after = re.replace_all(before, "$m/$d/$y");
assert_eq!(after, "03/14/2012, 01/01/2013 and 07/05/2014");
# }
```

If you wish to match against whitespace in this mode, you can still use `\s`,
`\n`, `\t`, etc. For escaping a single space character, you can escape it
directly with `\ `, use its hex character code `\x20` or temporarily disable
the `x` flag, e.g., `(?-x: )`.

# Example: match multiple regular expressions simultaneously

This demonstrates how to use a `RegexSet` to match multiple (possibly
overlapping) regular expressions in a single scan of the search text:

```rust
use regex::RegexSet;

let set = RegexSet::new(&[
    r"\w+",
    r"\d+",
    r"\pL+",
    r"foo",
    r"bar",
    r"barfoo",
    r"foobar",
]).unwrap();

// Iterate over and collect all of the matches.
let matches: Vec<_> = set.matches("foobar").into_iter().collect();
assert_eq!(matches, vec![0, 2, 3, 4, 6]);

// You can also test whether a particular regex matched:
let matches = set.matches("foobar");
assert!(!matches.matched(5));
assert!(matches.matched(6));
```

# Pay for what you use

With respect to searching text with a regular expression, there are three
questions that can be asked:

1. Does the text match this expression?
2. If so, where does it match?
3. Where did the capturing groups match?

Generally speaking, this crate could provide a function to answer only #3,
which would subsume #1 and #2 automatically. However, it can be significantly
more expensive to compute the location of capturing group matches, so it's best
not to do it if you don't need to.

Therefore, only use what you need. For example, don't use `find` if you
only need to test if an expression matches a string. (Use `is_match`
instead.)

# Unicode

This implementation executes regular expressions **only** on valid UTF-8
while exposing match locations as byte indices into the search string. (To
relax this restriction, use the [`bytes`](bytes/index.html) sub-module.)

Only simple case folding is supported. Namely, when matching
case-insensitively, the characters are first mapped using the "simple" case
folding rules defined by Unicode.

Regular expressions themselves are **only** interpreted as a sequence of
Unicode scalar values. This means you can use Unicode characters directly
in your expression:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"(?i)Δ+").unwrap();
let mat = re.find("ΔδΔ").unwrap();
assert_eq!((mat.start(), mat.end()), (0, 6));
# }
```

Most features of the regular expressions in this crate are Unicode aware. Here
are some examples:

* `.` will match any valid UTF-8 encoded Unicode scalar value except for `\n`.
  (To also match `\n`, enable the `s` flag, e.g., `(?s:.)`.)
* `\w`, `\d` and `\s` are Unicode aware. For example, `\s` will match all forms
  of whitespace categorized by Unicode.
* `\b` matches a Unicode word boundary.
* Negated character classes like `[^a]` match all Unicode scalar values except
  for `a`.
* `^` and `$` are **not** Unicode aware in multi-line mode. Namely, they only
  recognize `\n` and not any of the other forms of line terminators defined
  by Unicode.

Unicode general categories, scripts, script extensions, ages and a smattering
of boolean properties are available as character classes. For example, you can
match a sequence of numerals, Greek or Cherokee letters:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"[\pN\p{Greek}\p{Cherokee}]+").unwrap();
let mat = re.find("abcΔᎠβⅠᏴγδⅡxyz").unwrap();
assert_eq!((mat.start(), mat.end()), (3, 23));
# }
```

For a more detailed breakdown of Unicode support with respect to
[UTS#18](http://unicode.org/reports/tr18/),
please see the
[UNICODE](https://github.com/rust-lang/regex/blob/master/UNICODE.md)
document in the root of the regex repository.

# Opt out of Unicode support

The `bytes` sub-module provides a `Regex` type that can be used to match
on `&[u8]`. By default, text is interpreted as UTF-8 just like it is with
the main `Regex` type. However, this behavior can be disabled by turning
off the `u` flag, even if doing so could result in matching invalid UTF-8.
For example, when the `u` flag is disabled, `.` will match any byte instead
of any Unicode scalar value.

Disabling the `u` flag is also possible with the standard `&str`-based `Regex`
type, but it is only allowed where the UTF-8 invariant is maintained. For
example, `(?-u:\w)` is an ASCII-only `\w` character class and is legal in an
`&str`-based `Regex`, but `(?-u:\xFF)` will attempt to match the raw byte
`\xFF`, which is invalid UTF-8 and therefore is illegal in `&str`-based
regexes.

Finally, since Unicode support requires bundling large Unicode data
tables, this crate exposes knobs to disable the compilation of those
data tables, which can be useful for shrinking binary size and reducing
compilation times. For details on how to do that, see the section on [crate
features](#crate-features).

# Syntax

The syntax supported in this crate is documented below.

Note that the regular expression parser and abstract syntax are exposed in
a separate crate, [`regex-syntax`](https://docs.rs/regex-syntax).

## Matching one character

<pre class="rust">
.             any character except new line (includes new line with s flag)
\d            digit (\p{Nd})
\D            not digit
\pN           One-letter name Unicode character class
\p{Greek}     Unicode character class (general category or script)
\PN           Negated one-letter name Unicode character class
\P{Greek}     negated Unicode character class (general category or script)
</pre>

### Character classes

<pre class="rust">
[xyz]         A character class matching either x, y or z (union).
[^xyz]        A character class matching any character except x, y and z.
[a-z]         A character class matching any character in range a-z.
[[:alpha:]]   ASCII character class ([A-Za-z])
[[:^alpha:]]  Negated ASCII character class ([^A-Za-z])
[x[^xyz]]     Nested/grouping character class (matching any character except y and z)
[a-y&&xyz]    Intersection (matching x or y)
[0-9&&[^4]]   Subtraction using intersection and negation (matching 0-9 except 4)
[0-9--4]      Direct subtraction (matching 0-9 except 4)
[a-g~~b-h]    Symmetric difference (matching `a` and `h` only)
[\[\]]        Escaping in character classes (matching [ or ])
</pre>

Any named character class may appear inside a bracketed `[...]` character
class. For example, `[\p{Greek}[:digit:]]` matches any Greek or ASCII
digit. `[\p{Greek}&&\pL]` matches Greek letters.

Precedence in character classes, from most binding to least:

1. Ranges: `a-cd` == `[a-c]d`
2. Union: `ab&&bc` == `[ab]&&[bc]`
3. Intersection: `^a-z&&b` == `^[a-z&&b]`
4. Negation

## Composites

<pre class="rust">
xy    concatenation (x followed by y)
x|y   alternation (x or y, prefer x)
</pre>

## Repetitions

<pre class="rust">
x*        zero or more of x (greedy)
x+        one or more of x (greedy)
x?        zero or one of x (greedy)
x*?       zero or more of x (ungreedy/lazy)
x+?       one or more of x (ungreedy/lazy)
x??       zero or one of x (ungreedy/lazy)
x{n,m}    at least n x and at most m x (greedy)
x{n,}     at least n x (greedy)
x{n}      exactly n x
x{n,m}?   at least n x and at most m x (ungreedy/lazy)
x{n,}?    at least n x (ungreedy/lazy)
x{n}?     exactly n x
</pre>

## Empty matches

<pre class="rust">
^     the beginning of text (or start-of-line with multi-line mode)
$     the end of text (or end-of-line with multi-line mode)
\A    only the beginning of text (even with multi-line mode enabled)
\z    only the end of text (even with multi-line mode enabled)
\b    a Unicode word boundary (\w on one side and \W, \A, or \z on other)
\B    not a Unicode word boundary
</pre>

## Grouping and flags

<pre class="rust">
(exp)          numbered capture group (indexed by opening parenthesis)
(?P&lt;name&gt;exp)  named (also numbered) capture group (allowed chars: [_0-9a-zA-Z.\[\]])
(?:exp)        non-capturing group
(?flags)       set flags within current group
(?flags:exp)   set flags for exp (non-capturing)
</pre>

Flags are each a single character. For example, `(?x)` sets the flag `x`
and `(?-x)` clears the flag `x`. Multiple flags can be set or cleared at
the same time: `(?xy)` sets both the `x` and `y` flags and `(?x-y)` sets
the `x` flag and clears the `y` flag.

All flags are by default disabled unless stated otherwise. They are:

<pre class="rust">
i     case-insensitive: letters match both upper and lower case
m     multi-line mode: ^ and $ match begin/end of line
s     allow . to match \n
U     swap the meaning of x* and x*?
u     Unicode support (enabled by default)
x     ignore whitespace and allow line comments (starting with `#`)
</pre>

Flags can be toggled within a pattern. Here's an example that matches
case-insensitively for the first part but case-sensitively for the second part:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"(?i)a+(?-i)b+").unwrap();
let cap = re.captures("AaAaAbbBBBb").unwrap();
assert_eq!(&cap[0], "AaAaAbb");
# }
```

Notice that the `a+` matches either `a` or `A`, but the `b+` only matches
`b`.

Multi-line mode means `^` and `$` no longer match just at the beginning/end of
the input, but at the beginning/end of lines:

```
# use regex::Regex;
let re = Regex::new(r"(?m)^line \d+").unwrap();
let m = re.find("line one\nline 2\n").unwrap();
assert_eq!(m.as_str(), "line 2");
```

Note that `^` matches after new lines, even at the end of input:

```
# use regex::Regex;
let re = Regex::new(r"(?m)^").unwrap();
let m = re.find_iter("test\n").last().unwrap();
assert_eq!((m.start(), m.end()), (5, 5));
```

Here is an example that uses an ASCII word boundary instead of a Unicode
word boundary:

```rust
# extern crate regex; use regex::Regex;
# fn main() {
let re = Regex::new(r"(?-u:\b).+(?-u:\b)").unwrap();
let cap = re.captures("$$abc$$").unwrap();
assert_eq!(&cap[0], "abc");
# }
```

## Escape sequences

<pre class="rust">
\*          literal *, works for any punctuation character: \.+*?()|[]{}^$
\a          bell (\x07)
\f          form feed (\x0C)
\t          horizontal tab
\n          new line
\r          carriage return
\v          vertical tab (\x0B)
\123        octal character code (up to three digits) (when enabled)
\x7F        hex character code (exactly two digits)
\x{10FFFF}  any hex character code corresponding to a Unicode code point
\u007F      hex character code (exactly four digits)
\u{7F}      any hex character code corresponding to a Unicode code point
\U0000007F  hex character code (exactly eight digits)
\U{7F}      any hex character code corresponding to a Unicode code point
</pre>

## Perl character classes (Unicode friendly)

These classes are based on the definitions provided in
[UTS#18](http://www.unicode.org/reports/tr18/#Compatibility_Properties):

<pre class="rust">
\d     digit (\p{Nd})
\D     not digit
\s     whitespace (\p{White_Space})
\S     not whitespace
\w     word character (\p{Alphabetic} + \p{M} + \d + \p{Pc} + \p{Join_Control})
\W     not word character
</pre>

## ASCII character classes

<pre class="rust">
[[:alnum:]]    alphanumeric ([0-9A-Za-z])
[[:alpha:]]    alphabetic ([A-Za-z])
[[:ascii:]]    ASCII ([\x00-\x7F])
[[:blank:]]    blank ([\t ])
[[:cntrl:]]    control ([\x00-\x1F\x7F])
[[:digit:]]    digits ([0-9])
[[:graph:]]    graphical ([!-~])
[[:lower:]]    lower case ([a-z])
[[:print:]]    printable ([ -~])
[[:punct:]]    punctuation ([!-/:-@\[-`{-~])
[[:space:]]    whitespace ([\t\n\v\f\r ])
[[:upper:]]    upper case ([A-Z])
[[:word:]]     word characters ([0-9A-Za-z_])
[[:xdigit:]]   hex digit ([0-9A-Fa-f])
</pre>

# Crate features

By default, this crate tries pretty hard to make regex matching both as fast
as possible and as correct as it can be, within reason. This means that there
is a lot of code dedicated to performance, the handling of Unicode data and the
Unicode data itself. Overall, this leads to more dependencies, larger binaries
and longer compile times.  This trade off may not be appropriate in all cases,
and indeed, even when all Unicode and performance features are disabled, one
is still left with a perfectly serviceable regex engine that will work well
in many cases.

This crate exposes a number of features for controlling that trade off. Some
of these features are strictly performance oriented, such that disabling them
won't result in a loss of functionality, but may result in worse performance.
Other features, such as the ones controlling the presence or absence of Unicode
data, can result in a loss of functionality. For example, if one disables the
`unicode-case` feature (described below), then compiling the regex `(?i)a`
will fail since Unicode case insensitivity is enabled by default. Instead,
callers must use `(?i-u)a` instead to disable Unicode case folding. Stated
differently, enabling or disabling any of the features below can only add or
subtract from the total set of valid regular expressions. Enabling or disabling
a feature will never modify the match semantics of a regular expression.

All features below are enabled by default.

### Ecosystem features

* **std** -
  When enabled, this will cause `regex` to use the standard library. Currently,
  disabling this feature will always result in a compilation error. It is
  intended to add `alloc`-only support to regex in the future.

### Performance features

* **perf** -
  Enables all performance related features. This feature is enabled by default
  and will always cover all features that improve performance, even if more
  are added in the future.
* **perf-cache** -
  Enables the use of very fast thread safe caching for internal match state.
  When this is disabled, caching is still used, but with a slower and simpler
  implementation. Disabling this drops the `thread_local` and `lazy_static`
  dependencies.
* **perf-dfa** -
  Enables the use of a lazy DFA for matching. The lazy DFA is used to compile
  portions of a regex to a very fast DFA on an as-needed basis. This can
  result in substantial speedups, usually by an order of magnitude on large
  haystacks. The lazy DFA does not bring in any new dependencies, but it can
  make compile times longer.
* **perf-inline** -
  Enables the use of aggressive inlining inside match routines. This reduces
  the overhead of each match. The aggressive inlining, however, increases
  compile times and binary size.
* **perf-literal** -
  Enables the use of literal optimizations for speeding up matches. In some
  cases, literal optimizations can result in speedups of _several_ orders of
  magnitude. Disabling this drops the `aho-corasick` and `memchr` dependencies.

### Unicode features

* **unicode** -
  Enables all Unicode features. This feature is enabled by default, and will
  always cover all Unicode features, even if more are added in the future.
* **unicode-age** -
  Provide the data for the
  [Unicode `Age` property](https://www.unicode.org/reports/tr44/tr44-24.html#Character_Age).
  This makes it possible to use classes like `\p{Age:6.0}` to refer to all
  codepoints first introduced in Unicode 6.0
* **unicode-bool** -
  Provide the data for numerous Unicode boolean properties. The full list
  is not included here, but contains properties like `Alphabetic`, `Emoji`,
  `Lowercase`, `Math`, `Uppercase` and `White_Space`.
* **unicode-case** -
  Provide the data for case insensitive matching using
  [Unicode's "simple loose matches" specification](https://www.unicode.org/reports/tr18/#Simple_Loose_Matches).
* **unicode-gencat** -
  Provide the data for
  [Unicode general categories](https://www.unicode.org/reports/tr44/tr44-24.html#General_Category_Values).
  This includes, but is not limited to, `Decimal_Number`, `Letter`,
  `Math_Symbol`, `Number` and `Punctuation`.
* **unicode-perl** -
  Provide the data for supporting the Unicode-aware Perl character classes,
  corresponding to `\w`, `\s` and `\d`. This is also necessary for using
  Unicode-aware word boundary assertions. Note that if this feature is
  disabled, the `\s` and `\d` character classes are still available if the
  `unicode-bool` and `unicode-gencat` features are enabled, respectively.
* **unicode-script** -
  Provide the data for
  [Unicode scripts and script extensions](https://www.unicode.org/reports/tr24/).
  This includes, but is not limited to, `Arabic`, `Cyrillic`, `Hebrew`,
  `Latin` and `Thai`.
* **unicode-segment** -
  Provide the data necessary to provide the properties used to implement the
  [Unicode text segmentation algorithms](https://www.unicode.org/reports/tr29/).
  This enables using classes like `\p{gcb=Extend}`, `\p{wb=Katakana}` and
  `\p{sb=ATerm}`.


# Untrusted input

This crate can handle both untrusted regular expressions and untrusted
search text.

Untrusted regular expressions are handled by capping the size of a compiled
regular expression.
(See [`RegexBuilder::size_limit`](struct.RegexBuilder.html#method.size_limit).)
Without this, it would be trivial for an attacker to exhaust your system's
memory with expressions like `a{100}{100}{100}`.

Untrusted search text is allowed because the matching engine(s) in this
crate have time complexity `O(mn)` (with `m ~ regex` and `n ~ search
text`), which means there's no way to cause exponential blow-up like with
some other regular expression engines. (We pay for this by disallowing
features like arbitrary look-ahead and backreferences.)

When a DFA is used, pathological cases with exponential state blow-up are
avoided by constructing the DFA lazily or in an "online" manner. Therefore,
at most one new state can be created for each byte of input. This satisfies
our time complexity guarantees, but can lead to memory growth
proportional to the size of the input. As a stopgap, the DFA is only
allowed to store a fixed number of states. When the limit is reached, its
states are wiped and continues on, possibly duplicating previous work. If
the limit is reached too frequently, it gives up and hands control off to
another matching engine with fixed memory requirements.
(The DFA size limit can also be tweaked. See
[`RegexBuilder::dfa_size_limit`](struct.RegexBuilder.html#method.dfa_size_limit).)
*/

#![deny(missing_docs)]
#![cfg_attr(test, deny(warnings))]
#![cfg_attr(feature = "pattern", feature(pattern))]

#[cfg(not(feature = "std"))]
compile_error!("`std` feature is currently required to build this crate");

#[cfg(feature = "perf-literal")]
extern crate aho_corasick;
// #[cfg(doctest)]
// extern crate doc_comment;
#[cfg(feature = "perf-literal")]
extern crate memchr;
#[cfg(test)]
#[cfg_attr(feature = "perf-literal", macro_use)]
extern crate quickcheck;
extern crate regex_syntax as syntax;
#[cfg(feature = "perf-cache")]
extern crate thread_local;

// #[cfg(doctest)]
// doc_comment::doctest!("../README.md");

#[cfg(feature = "std")]
pub use error::Error;
#[cfg(feature = "std")]
pub use re_builder::set_unicode::*;
#[cfg(feature = "std")]
pub use re_builder::unicode::*;
#[cfg(feature = "std")]
pub use re_set::unicode::*;
#[cfg(feature = "std")]
#[cfg(feature = "std")]
pub use re_unicode::{
    escape, CaptureLocations, CaptureMatches, CaptureNames, Captures,
    Locations, Match, Matches, NoExpand, Regex, Replacer, ReplacerRef, Split,
    SplitN, SubCaptureMatches,
};

/**
Match regular expressions on arbitrary bytes.

This module provides a nearly identical API to the one found in the
top-level of this crate. There are two important differences:

1. Matching is done on `&[u8]` instead of `&str`. Additionally, `Vec<u8>`
is used where `String` would have been used.
2. Unicode support can be disabled even when disabling it would result in
matching invalid UTF-8 bytes.

# Example: match null terminated string

This shows how to find all null-terminated strings in a slice of bytes:

```rust
# use regex::bytes::Regex;
let re = Regex::new(r"(?-u)(?P<cstr>[^\x00]+)\x00").unwrap();
let text = b"foo\x00bar\x00baz\x00";

// Extract all of the strings without the null terminator from each match.
// The unwrap is OK here since a match requires the `cstr` capture to match.
let cstrs: Vec<&[u8]> =
    re.captures_iter(text)
      .map(|c| c.name("cstr").unwrap().as_bytes())
      .collect();
assert_eq!(vec![&b"foo"[..], &b"bar"[..], &b"baz"[..]], cstrs);
```

# Example: selectively enable Unicode support

This shows how to match an arbitrary byte pattern followed by a UTF-8 encoded
string (e.g., to extract a title from a Matroska file):

```rust
# use std::str;
# use regex::bytes::Regex;
let re = Regex::new(
    r"(?-u)\x7b\xa9(?:[\x80-\xfe]|[\x40-\xff].)(?u:(.*))"
).unwrap();
let text = b"\x12\xd0\x3b\x5f\x7b\xa9\x85\xe2\x98\x83\x80\x98\x54\x76\x68\x65";
let caps = re.captures(text).unwrap();

// Notice that despite the `.*` at the end, it will only match valid UTF-8
// because Unicode mode was enabled with the `u` flag. Without the `u` flag,
// the `.*` would match the rest of the bytes.
let mat = caps.get(1).unwrap();
assert_eq!((7, 10), (mat.start(), mat.end()));

// If there was a match, Unicode mode guarantees that `title` is valid UTF-8.
let title = str::from_utf8(&caps[1]).unwrap();
assert_eq!("☃", title);
```

In general, if the Unicode flag is enabled in a capture group and that capture
is part of the overall match, then the capture is *guaranteed* to be valid
UTF-8.

# Syntax

The supported syntax is pretty much the same as the syntax for Unicode
regular expressions with a few changes that make sense for matching arbitrary
bytes:

1. The `u` flag can be disabled even when disabling it might cause the regex to
match invalid UTF-8. When the `u` flag is disabled, the regex is said to be in
"ASCII compatible" mode.
2. In ASCII compatible mode, neither Unicode scalar values nor Unicode
character classes are allowed.
3. In ASCII compatible mode, Perl character classes (`\w`, `\d` and `\s`)
revert to their typical ASCII definition. `\w` maps to `[[:word:]]`, `\d` maps
to `[[:digit:]]` and `\s` maps to `[[:space:]]`.
4. In ASCII compatible mode, word boundaries use the ASCII compatible `\w` to
determine whether a byte is a word byte or not.
5. Hexadecimal notation can be used to specify arbitrary bytes instead of
Unicode codepoints. For example, in ASCII compatible mode, `\xFF` matches the
literal byte `\xFF`, while in Unicode mode, `\xFF` is a Unicode codepoint that
matches its UTF-8 encoding of `\xC3\xBF`. Similarly for octal notation when
enabled.
6. In ASCII compatible mode, `.` matches any *byte* except for `\n`. When the
`s` flag is additionally enabled, `.` matches any byte.

# Performance

In general, one should expect performance on `&[u8]` to be roughly similar to
performance on `&str`.
*/
#[cfg(feature = "std")]
pub mod bytes {
    pub use re_builder::bytes::*;
    pub use re_builder::set_bytes::*;
    pub use re_bytes::*;
    pub use re_set::bytes::*;
}

mod backtrack;
mod cache;
mod compile;
#[cfg(feature = "perf-dfa")]
mod dfa;
mod error;
mod exec;
mod expand;
mod find_byte;
#[cfg(feature = "perf-literal")]
mod freqs;
mod input;
mod literal;
#[cfg(feature = "pattern")]
mod pattern;
mod pikevm;
mod prog;
mod re_builder;
mod re_bytes;
mod re_set;
mod re_trait;
mod re_unicode;
mod sparse;
mod utf8;

/// The `internal` module exists to support suspicious activity, such as
/// testing different matching engines and supporting the `regex-debug` CLI
/// utility.
#[doc(hidden)]
#[cfg(feature = "std")]
pub mod internal {
    pub use compile::Compiler;
    pub use exec::{Exec, ExecBuilder};
    pub use input::{Char, CharInput, Input, InputAt};
    pub use literal::LiteralSearcher;
    pub use prog::{EmptyLook, Inst, InstRanges, Program};
}
#![no_std]
#![no_main]

//extern crate panic_halt;

use riscv_rt::entry;
use hifive1::hal::prelude::*;
use hifive1::hal::spi::{Spi, MODE_0, SpiX};
use hifive1::hal::gpio::{gpio0::Pin10, Input, Floating};
use hifive1::hal::delay::Delay;
use hifive1::hal::clock::Clocks;
use hifive1::hal::DeviceResources;
use hifive1::{sprintln, pin};
use core::panic::PanicInfo;
use embedded_hal::blocking::delay::DelayUs;
use embedded_hal::blocking::spi::WriteIter;

#[inline(never)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    sprintln!("panic: {}", info);
    loop {
        use core::sync::atomic;
        use core::sync::atomic::Ordering;
        atomic::compiler_fence(Ordering::SeqCst);
    }
}

#[derive(Debug)]
enum EspError {
    ProtocolError,
    BufferOverflow,
    WouldBlock
}

struct EspWiFi<SPI, PINS> {
    spi: Spi<SPI, PINS>,
    handshake: Pin10<Input<Floating>>,
    delay: FastDelay,
}

impl<SPI: SpiX, PINS> EspWiFi<SPI, PINS> {
    fn send_bytes(&mut self, bytes: &[u8]) {
        self.delay.delay_us(18u32);
        self.spi.write(bytes).unwrap();
        self.delay.delay_us(5000u32);
    }

    fn transfer(&mut self, buffer: &mut [u8]) {
        self.delay.delay_us(18u32);
        self.spi.transfer(buffer).unwrap();
        self.delay.delay_us(5000u32);
    }

    fn discard(&mut self, size: usize) {
        self.delay.delay_us(18u32);
        self.spi.write_iter((0..size).map(|_| 0x00)).unwrap();
        self.delay.delay_us(5000u32);
    }

    pub fn send(&mut self, s: &str) {
        let bytes = s.as_bytes();
        assert!(bytes.len() <= 127);

        self.send_bytes(&[0x02, 0x00, 0x00, 0x00]);
        self.send_bytes(&[bytes.len() as u8, 0x00, 0x00, 0x41]);
        self.send_bytes(bytes);
    }

    pub fn recv<'a>(&mut self, buffer: &'a mut [u8]) -> Result<&'a str, EspError> {
        if self.handshake.is_low().unwrap() {
            return Err(EspError::WouldBlock);
        }

        self.send_bytes(&[0x01, 0x00, 0x00, 0x00]);

        let mut request = [0u8; 4];
        self.transfer(&mut request);
        if request[3] != 0x42 {
            return Err(EspError::ProtocolError);
        }

        let n = (request[0] & 0x7F) as usize + ((request[1] as usize) << 7);
        if n > buffer.len() {
            self.discard(n);
            return Err(EspError::BufferOverflow);
        }

        self.transfer(&mut buffer[..n]);
        Ok(core::str::from_utf8(&buffer[..n]).unwrap())
    }
}

struct FastDelay {
    us_cycles: u64,
}

impl FastDelay {
    pub fn new(clocks: Clocks) -> Self {
        Self {
            us_cycles: clocks.coreclk().0 as u64 * 3 / 2_000_000,
        }
    }
}

impl DelayUs<u32> for FastDelay {
    fn delay_us(&mut self, us: u32) {
        use riscv::register::mcycle;

        let t = mcycle::read64() + self.us_cycles * (us as u64);
        while mcycle::read64() < t {}
    }
}

#[entry]
fn main() -> ! {
    let dr = DeviceResources::take().unwrap();
    let p = dr.peripherals;
    let gpio = dr.pins;

    // Configure clocks
    let clocks = hifive1::clock::configure(p.PRCI, p.AONCLK, 320.mhz().into());

    // Configure UART for stdout
    hifive1::stdout::configure(p.UART0, pin!(gpio, uart0_tx), pin!(gpio, uart0_rx), 115_200.bps(), clocks);

    // Configure SPI pins
    let mosi = pin!(gpio, spi0_mosi).into_iof0();
    let miso = pin!(gpio, spi0_miso).into_iof0();
    let sck = pin!(gpio, spi0_sck).into_iof0();
    let cs = pin!(gpio, spi0_ss2).into_iof0();

    // Configure SPI
    let pins = (mosi, miso, sck, cs);
    let spi = Spi::new(p.QSPI1, pins, MODE_0, 100_000.hz(), clocks);
    let mem = Spi::new(p.QSPI0, (), MODE_0, 100_000.hz(), clocks);

    let handshake = gpio.pin10.into_floating_input();
    let mut wifi = EspWiFi {
        spi,
        handshake,
        delay: FastDelay::new(clocks),
    };

    sprintln!("WiFi Test !!");

    Delay.delay_ms(10u32);

    let mut buffer = [0u8; 256];

    wifi.send("AT+CWMODE=2\r\n");
    Delay.delay_ms(20u32);
    sprintln!("resp: {:?}", wifi.recv(&mut buffer));
  //Delay.delay_ms(20u32);
  //wifi.send("AT+CWJAP=\"The Theriaults\",\"Buddydog1@\" \r\n");
  //sprintln!("resp: {:?}", wifi.recv(&mut buffer));

    loop {
    }
}
use std::io::{self, Write};
#[cfg(feature = "paging")]
use std::process::Child;

use crate::error::*;
#[cfg(feature = "paging")]
use crate::less::retrieve_less_version;
#[cfg(feature = "paging")]
use crate::paging::PagingMode;
#[cfg(feature = "paging")]
use crate::wrapping::WrappingMode;

#[cfg(feature = "paging")]
#[derive(Debug, PartialEq)]
enum SingleScreenAction {
    Quit,
    Nothing,
}

#[derive(Debug)]
pub enum OutputType {
    #[cfg(feature = "paging")]
    Pager(Child),
    Stdout(io::Stdout),
}

impl OutputType {
    #[cfg(feature = "paging")]
    pub fn from_mode(
        paging_mode: PagingMode,
        wrapping_mode: WrappingMode,
        pager: Option<&str>,
    ) -> Result<Self> {
        use self::PagingMode::*;
        Ok(match paging_mode {
            Always => OutputType::try_pager(SingleScreenAction::Nothing, wrapping_mode, pager)?,
            QuitIfOneScreen => {
                OutputType::try_pager(SingleScreenAction::Quit, wrapping_mode, pager)?
            }
            _ => OutputType::stdout(),
        })
    }

    /// Try to launch the pager. Fall back to stdout in case of errors.
    #[cfg(feature = "paging")]
    fn try_pager(
        single_screen_action: SingleScreenAction,
        wrapping_mode: WrappingMode,
        pager_from_config: Option<&str>,
    ) -> Result<Self> {
        use crate::pager::{self, PagerKind, PagerSource};
        use std::process::{Command, Stdio};

        let pager_opt =
            pager::get_pager(pager_from_config).map_err(|_| "Could not parse pager command.")?;

        let pager = match pager_opt {
            Some(pager) => pager,
            None => return Ok(OutputType::stdout()),
        };

        if pager.kind == PagerKind::Bat {
            return Err(Error::InvalidPagerValueBat);
        }

        let resolved_path = match grep_cli::resolve_binary(&pager.bin) {
            Ok(path) => path,
            Err(_) => {
                return Ok(OutputType::stdout());
            }
        };

        let mut p = Command::new(resolved_path);
        let args = pager.args;

        if pager.kind == PagerKind::Less {
            // less needs to be called with the '-R' option in order to properly interpret the
            // ANSI color sequences printed by bat. If someone has set PAGER="less -F", we
            // therefore need to overwrite the arguments and add '-R'.
            //
            // We only do this for PAGER (as it is not specific to 'bat'), not for BAT_PAGER
            // or bats '--pager' command line option.
            let replace_arguments_to_less = pager.source == PagerSource::EnvVarPager;

            if args.is_empty() || replace_arguments_to_less {
                p.arg("--RAW-CONTROL-CHARS");
                if single_screen_action == SingleScreenAction::Quit {
                    p.arg("--quit-if-one-screen");
                }

                if wrapping_mode == WrappingMode::NoWrapping(true) {
                    p.arg("--chop-long-lines");
                }

                // Passing '--no-init' fixes a bug with '--quit-if-one-screen' in older
                // versions of 'less'. Unfortunately, it also breaks mouse-wheel support.
                //
                // See: http://www.greenwoodsoftware.com/less/news.530.html
                //
                // For newer versions (530 or 558 on Windows), we omit '--no-init' as it
                // is not needed anymore.
                match retrieve_less_version(&pager.bin) {
                    None => {
                        p.arg("--no-init");
                    }
                    Some(version) if (version < 530 || (cfg!(windows) && version < 558)) => {
                        p.arg("--no-init");
                    }
                    _ => {}
                }
            } else {
                p.args(args);
            }
            p.env("LESSCHARSET", "UTF-8");
        } else {
            p.args(args);
        };

        Ok(p.stdin(Stdio::piped())
            .spawn()
            .map(OutputType::Pager)
            .unwrap_or_else(|_| OutputType::stdout()))
    }

    pub(crate) fn stdout() -> Self {
        OutputType::Stdout(io::stdout())
    }

    #[cfg(feature = "paging")]
    pub(crate) fn is_pager(&self) -> bool {
        matches!(self, OutputType::Pager(_))
    }

    #[cfg(not(feature = "paging"))]
    pub(crate) fn is_pager(&self) -> bool {
        false
    }

    pub fn handle(&mut self) -> Result<&mut dyn Write> {
        Ok(match *self {
            #[cfg(feature = "paging")]
            OutputType::Pager(ref mut command) => command
                .stdin
                .as_mut()
                .ok_or("Could not open stdin for pager")?,
            OutputType::Stdout(ref mut handle) => handle,
        })
    }
}

#[cfg(feature = "paging")]
impl Drop for OutputType {
    fn drop(&mut self) {
        if let OutputType::Pager(ref mut command) = *self {
            let _ = command.wait();
        }
    }
}
#[macro_export]
macro_rules! bat_warning {
    ($($arg:tt)*) => ({
        use ansi_term::Colour::Yellow;
        eprintln!("{}: {}", Yellow.paint("[bat warning]"), format!($($arg)*));
    })
}
/*!
The DFA matching engine.

A DFA provides faster matching because the engine is in exactly one state at
any point in time. In the NFA, there may be multiple active states, and
considerable CPU cycles are spent shuffling them around. In finite automata
speak, the DFA follows epsilon transitions in the regex far less than the NFA.

A DFA is a classic trade off between time and space. The NFA is slower, but
its memory requirements are typically small and predictable. The DFA is faster,
but given the right regex and the right input, the number of states in the
DFA can grow exponentially. To mitigate this space problem, we do two things:

1. We implement an *online* DFA. That is, the DFA is constructed from the NFA
   during a search. When a new state is computed, it is stored in a cache so
   that it may be reused. An important consequence of this implementation
   is that states that are never reached for a particular input are never
   computed. (This is impossible in an "offline" DFA which needs to compute
   all possible states up front.)
2. If the cache gets too big, we wipe it and continue matching.

In pathological cases, a new state can be created for every byte of input.
(e.g., The regex `(a|b)*a(a|b){20}` on a long sequence of a's and b's.)
In this case, performance regresses to slightly slower than the full NFA
simulation, in large part because the cache becomes useless. If the cache
is wiped too frequently, the DFA quits and control falls back to one of the
NFA simulations.

Because of the "lazy" nature of this DFA, the inner matching loop is
considerably more complex than one might expect out of a DFA. A number of
tricks are employed to make it fast. Tread carefully.

N.B. While this implementation is heavily commented, Russ Cox's series of
articles on regexes is strongly recommended: https://swtch.com/~rsc/regexp/
(As is the DFA implementation in RE2, which heavily influenced this
implementation.)
*/

use std::collections::HashMap;
use std::fmt;
use std::iter::repeat;
use std::mem;
use std::sync::Arc;

use exec::ProgramCache;
use prog::{Inst, Program};
use sparse::SparseSet;

/// Return true if and only if the given program can be executed by a DFA.
///
/// Generally, a DFA is always possible. A pathological case where it is not
/// possible is if the number of NFA states exceeds `u32::MAX`, in which case,
/// this function will return false.
///
/// This function will also return false if the given program has any Unicode
/// instructions (Char or Ranges) since the DFA operates on bytes only.
pub fn can_exec(insts: &Program) -> bool {
    use prog::Inst::*;
    // If for some reason we manage to allocate a regex program with more
    // than i32::MAX instructions, then we can't execute the DFA because we
    // use 32 bit instruction pointer deltas for memory savings.
    // If i32::MAX is the largest positive delta,
    // then -i32::MAX == i32::MIN + 1 is the largest negative delta,
    // and we are OK to use 32 bits.
    if insts.dfa_size_limit == 0 || insts.len() > ::std::i32::MAX as usize {
        return false;
    }
    for inst in insts {
        match *inst {
            Char(_) | Ranges(_) => return false,
            EmptyLook(_) | Match(_) | Save(_) | Split(_) | Bytes(_) => {}
        }
    }
    true
}

/// A reusable cache of DFA states.
///
/// This cache is reused between multiple invocations of the same regex
/// program. (It is not shared simultaneously between threads. If there is
/// contention, then new caches are created.)
#[derive(Debug)]
pub struct Cache {
    /// Group persistent DFA related cache state together. The sparse sets
    /// listed below are used as scratch space while computing uncached states.
    inner: CacheInner,
    /// qcur and qnext are ordered sets with constant time
    /// addition/membership/clearing-whole-set and linear time iteration. They
    /// are used to manage the sets of NFA states in DFA states when computing
    /// cached DFA states. In particular, the order of the NFA states matters
    /// for leftmost-first style matching. Namely, when computing a cached
    /// state, the set of NFA states stops growing as soon as the first Match
    /// instruction is observed.
    qcur: SparseSet,
    qnext: SparseSet,
}

/// `CacheInner` is logically just a part of Cache, but groups together fields
/// that aren't passed as function parameters throughout search. (This split
/// is mostly an artifact of the borrow checker. It is happily paid.)
#[derive(Debug)]
struct CacheInner {
    /// A cache of pre-compiled DFA states, keyed by the set of NFA states
    /// and the set of empty-width flags set at the byte in the input when the
    /// state was observed.
    ///
    /// A StatePtr is effectively a `*State`, but to avoid various inconvenient
    /// things, we just pass indexes around manually. The performance impact of
    /// this is probably an instruction or two in the inner loop. However, on
    /// 64 bit, each StatePtr is half the size of a *State.
    compiled: StateMap,
    /// The transition table.
    ///
    /// The transition table is laid out in row-major order, where states are
    /// rows and the transitions for each state are columns. At a high level,
    /// given state `s` and byte `b`, the next state can be found at index
    /// `s * 256 + b`.
    ///
    /// This is, of course, a lie. A StatePtr is actually a pointer to the
    /// *start* of a row in this table. When indexing in the DFA's inner loop,
    /// this removes the need to multiply the StatePtr by the stride. Yes, it
    /// matters. This reduces the number of states we can store, but: the
    /// stride is rarely 256 since we define transitions in terms of
    /// *equivalence classes* of bytes. Each class corresponds to a set of
    /// bytes that never discriminate a distinct path through the DFA from each
    /// other.
    trans: Transitions,
    /// A set of cached start states, which are limited to the number of
    /// permutations of flags set just before the initial byte of input. (The
    /// index into this vec is a `EmptyFlags`.)
    ///
    /// N.B. A start state can be "dead" (i.e., no possible match), so we
    /// represent it with a StatePtr.
    start_states: Vec<StatePtr>,
    /// Stack scratch space used to follow epsilon transitions in the NFA.
    /// (This permits us to avoid recursion.)
    ///
    /// The maximum stack size is the number of NFA states.
    stack: Vec<InstPtr>,
    /// The total number of times this cache has been flushed by the DFA
    /// because of space constraints.
    flush_count: u64,
    /// The total heap size of the DFA's cache. We use this to determine when
    /// we should flush the cache.
    size: usize,
    /// Scratch space used when building instruction pointer lists for new
    /// states. This helps amortize allocation.
    insts_scratch_space: Vec<u8>,
}

/// The transition table.
///
/// It is laid out in row-major order, with states as rows and byte class
/// transitions as columns.
///
/// The transition table is responsible for producing valid `StatePtrs`. A
/// `StatePtr` points to the start of a particular row in this table. When
/// indexing to find the next state this allows us to avoid a multiplication
/// when computing an index into the table.
#[derive(Clone)]
struct Transitions {
    /// The table.
    table: Vec<StatePtr>,
    /// The stride.
    num_byte_classes: usize,
}

/// Fsm encapsulates the actual execution of the DFA.
#[derive(Debug)]
pub struct Fsm<'a> {
    /// prog contains the NFA instruction opcodes. DFA execution uses either
    /// the `dfa` instructions or the `dfa_reverse` instructions from
    /// `exec::ExecReadOnly`. (It never uses `ExecReadOnly.nfa`, which may have
    /// Unicode opcodes that cannot be executed by the DFA.)
    prog: &'a Program,
    /// The start state. We record it here because the pointer may change
    /// when the cache is wiped.
    start: StatePtr,
    /// The current position in the input.
    at: usize,
    /// Should we quit after seeing the first match? e.g., When the caller
    /// uses `is_match` or `shortest_match`.
    quit_after_match: bool,
    /// The last state that matched.
    ///
    /// When no match has occurred, this is set to STATE_UNKNOWN.
    ///
    /// This is only useful when matching regex sets. The last match state
    /// is useful because it contains all of the match instructions seen,
    /// thereby allowing us to enumerate which regexes in the set matched.
    last_match_si: StatePtr,
    /// The input position of the last cache flush. We use this to determine
    /// if we're thrashing in the cache too often. If so, the DFA quits so
    /// that we can fall back to the NFA algorithm.
    last_cache_flush: usize,
    /// All cached DFA information that is persisted between searches.
    cache: &'a mut CacheInner,
}

/// The result of running the DFA.
///
/// Generally, the result is either a match or not a match, but sometimes the
/// DFA runs too slowly because the cache size is too small. In that case, it
/// gives up with the intent of falling back to the NFA algorithm.
///
/// The DFA can also give up if it runs out of room to create new states, or if
/// it sees non-ASCII bytes in the presence of a Unicode word boundary.
#[derive(Clone, Debug)]
pub enum Result<T> {
    Match(T),
    NoMatch(usize),
    Quit,
}

impl<T> Result<T> {
    /// Returns true if this result corresponds to a match.
    pub fn is_match(&self) -> bool {
        match *self {
            Result::Match(_) => true,
            Result::NoMatch(_) | Result::Quit => false,
        }
    }

    /// Maps the given function onto T and returns the result.
    ///
    /// If this isn't a match, then this is a no-op.
    #[cfg(feature = "perf-literal")]
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Result<U> {
        match self {
            Result::Match(t) => Result::Match(f(t)),
            Result::NoMatch(x) => Result::NoMatch(x),
            Result::Quit => Result::Quit,
        }
    }

    /// Sets the non-match position.
    ///
    /// If this isn't a non-match, then this is a no-op.
    fn set_non_match(self, at: usize) -> Result<T> {
        match self {
            Result::NoMatch(_) => Result::NoMatch(at),
            r => r,
        }
    }
}

/// `State` is a DFA state. It contains an ordered set of NFA states (not
/// necessarily complete) and a smattering of flags.
///
/// The flags are packed into the first byte of data.
///
/// States don't carry their transitions. Instead, transitions are stored in
/// a single row-major table.
///
/// Delta encoding is used to store the instruction pointers.
/// The first instruction pointer is stored directly starting
/// at data[1], and each following pointer is stored as an offset
/// to the previous one. If a delta is in the range -127..127,
/// it is packed into a single byte; Otherwise the byte 128 (-128 as an i8)
/// is coded as a flag, followed by 4 bytes encoding the delta.
#[derive(Clone, Eq, Hash, PartialEq)]
struct State {
    data: Arc<[u8]>,
}

/// `InstPtr` is a 32 bit pointer into a sequence of opcodes (i.e., it indexes
/// an NFA state).
///
/// Throughout this library, this is usually set to `usize`, but we force a
/// `u32` here for the DFA to save on space.
type InstPtr = u32;

/// Adds ip to data using delta encoding with respect to prev.
///
/// After completion, `data` will contain `ip` and `prev` will be set to `ip`.
fn push_inst_ptr(data: &mut Vec<u8>, prev: &mut InstPtr, ip: InstPtr) {
    let delta = (ip as i32) - (*prev as i32);
    write_vari32(data, delta);
    *prev = ip;
}

struct InstPtrs<'a> {
    base: usize,
    data: &'a [u8],
}

impl<'a> Iterator for InstPtrs<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.data.is_empty() {
            return None;
        }
        let (delta, nread) = read_vari32(self.data);
        let base = self.base as i32 + delta;
        debug_assert!(base >= 0);
        debug_assert!(nread > 0);
        self.data = &self.data[nread..];
        self.base = base as usize;
        Some(self.base)
    }
}

impl State {
    fn flags(&self) -> StateFlags {
        StateFlags(self.data[0])
    }

    fn inst_ptrs(&self) -> InstPtrs {
        InstPtrs { base: 0, data: &self.data[1..] }
    }
}

/// `StatePtr` is a 32 bit pointer to the start of a row in the transition
/// table.
///
/// It has many special values. There are two types of special values:
/// sentinels and flags.
///
/// Sentinels corresponds to special states that carry some kind of
/// significance. There are three such states: unknown, dead and quit states.
///
/// Unknown states are states that haven't been computed yet. They indicate
/// that a transition should be filled in that points to either an existing
/// cached state or a new state altogether. In general, an unknown state means
/// "follow the NFA's epsilon transitions."
///
/// Dead states are states that can never lead to a match, no matter what
/// subsequent input is observed. This means that the DFA should quit
/// immediately and return the longest match it has found thus far.
///
/// Quit states are states that imply the DFA is not capable of matching the
/// regex correctly. Currently, this is only used when a Unicode word boundary
/// exists in the regex *and* a non-ASCII byte is observed.
///
/// The other type of state pointer is a state pointer with special flag bits.
/// There are two flags: a start flag and a match flag. The lower bits of both
/// kinds always contain a "valid" `StatePtr` (indicated by the `STATE_MAX`
/// mask).
///
/// The start flag means that the state is a start state, and therefore may be
/// subject to special prefix scanning optimizations.
///
/// The match flag means that the state is a match state, and therefore the
/// current position in the input (while searching) should be recorded.
///
/// The above exists mostly in the service of making the inner loop fast.
/// In particular, the inner *inner* loop looks something like this:
///
/// ```ignore
/// while state <= STATE_MAX and i < len(text):
///     state = state.next[i]
/// ```
///
/// This is nice because it lets us execute a lazy DFA as if it were an
/// entirely offline DFA (i.e., with very few instructions). The loop will
/// quit only when we need to examine a case that needs special attention.
type StatePtr = u32;

/// An unknown state means that the state has not been computed yet, and that
/// the only way to progress is to compute it.
const STATE_UNKNOWN: StatePtr = 1 << 31;

/// A dead state means that the state has been computed and it is known that
/// once it is entered, no future match can ever occur.
const STATE_DEAD: StatePtr = STATE_UNKNOWN + 1;

/// A quit state means that the DFA came across some input that it doesn't
/// know how to process correctly. The DFA should quit and another matching
/// engine should be run in its place.
const STATE_QUIT: StatePtr = STATE_DEAD + 1;

/// A start state is a state that the DFA can start in.
///
/// Note that start states have their lower bits set to a state pointer.
const STATE_START: StatePtr = 1 << 30;

/// A match state means that the regex has successfully matched.
///
/// Note that match states have their lower bits set to a state pointer.
const STATE_MATCH: StatePtr = 1 << 29;

/// The maximum state pointer. This is useful to mask out the "valid" state
/// pointer from a state with the "start" or "match" bits set.
///
/// It doesn't make sense to use this with unknown, dead or quit state
/// pointers, since those pointers are sentinels and never have their lower
/// bits set to anything meaningful.
const STATE_MAX: StatePtr = STATE_MATCH - 1;

/// Byte is a u8 in spirit, but a u16 in practice so that we can represent the
/// special EOF sentinel value.
#[derive(Copy, Clone, Debug)]
struct Byte(u16);

/// A set of flags for zero-width assertions.
#[derive(Clone, Copy, Eq, Debug, Default, Hash, PartialEq)]
struct EmptyFlags {
    start: bool,
    end: bool,
    start_line: bool,
    end_line: bool,
    word_boundary: bool,
    not_word_boundary: bool,
}

/// A set of flags describing various configurations of a DFA state. This is
/// represented by a `u8` so that it is compact.
#[derive(Clone, Copy, Eq, Default, Hash, PartialEq)]
struct StateFlags(u8);

impl Cache {
    /// Create new empty cache for the DFA engine.
    pub fn new(prog: &Program) -> Self {
        // We add 1 to account for the special EOF byte.
        let num_byte_classes = (prog.byte_classes[255] as usize + 1) + 1;
        let starts = vec![STATE_UNKNOWN; 256];
        let mut cache = Cache {
            inner: CacheInner {
                compiled: StateMap::new(num_byte_classes),
                trans: Transitions::new(num_byte_classes),
                start_states: starts,
                stack: vec![],
                flush_count: 0,
                size: 0,
                insts_scratch_space: vec![],
            },
            qcur: SparseSet::new(prog.insts.len()),
            qnext: SparseSet::new(prog.insts.len()),
        };
        cache.inner.reset_size();
        cache
    }
}

impl CacheInner {
    /// Resets the cache size to account for fixed costs, such as the program
    /// and stack sizes.
    fn reset_size(&mut self) {
        self.size = (self.start_states.len() * mem::size_of::<StatePtr>())
            + (self.stack.len() * mem::size_of::<InstPtr>());
    }
}

impl<'a> Fsm<'a> {
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn forward(
        prog: &'a Program,
        cache: &ProgramCache,
        quit_after_match: bool,
        text: &[u8],
        at: usize,
    ) -> Result<usize> {
        let mut cache = cache.borrow_mut();
        let cache = &mut cache.dfa;
        let mut dfa = Fsm {
            prog: prog,
            start: 0, // filled in below
            at: at,
            quit_after_match: quit_after_match,
            last_match_si: STATE_UNKNOWN,
            last_cache_flush: at,
            cache: &mut cache.inner,
        };
        let (empty_flags, state_flags) = dfa.start_flags(text, at);
        dfa.start =
            match dfa.start_state(&mut cache.qcur, empty_flags, state_flags) {
                None => return Result::Quit,
                Some(STATE_DEAD) => return Result::NoMatch(at),
                Some(si) => si,
            };
        debug_assert!(dfa.start != STATE_UNKNOWN);
        dfa.exec_at(&mut cache.qcur, &mut cache.qnext, text)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn reverse(
        prog: &'a Program,
        cache: &ProgramCache,
        quit_after_match: bool,
        text: &[u8],
        at: usize,
    ) -> Result<usize> {
        let mut cache = cache.borrow_mut();
        let cache = &mut cache.dfa_reverse;
        let mut dfa = Fsm {
            prog: prog,
            start: 0, // filled in below
            at: at,
            quit_after_match: quit_after_match,
            last_match_si: STATE_UNKNOWN,
            last_cache_flush: at,
            cache: &mut cache.inner,
        };
        let (empty_flags, state_flags) = dfa.start_flags_reverse(text, at);
        dfa.start =
            match dfa.start_state(&mut cache.qcur, empty_flags, state_flags) {
                None => return Result::Quit,
                Some(STATE_DEAD) => return Result::NoMatch(at),
                Some(si) => si,
            };
        debug_assert!(dfa.start != STATE_UNKNOWN);
        dfa.exec_at_reverse(&mut cache.qcur, &mut cache.qnext, text)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn forward_many(
        prog: &'a Program,
        cache: &ProgramCache,
        matches: &mut [bool],
        text: &[u8],
        at: usize,
    ) -> Result<usize> {
        debug_assert!(matches.len() == prog.matches.len());
        let mut cache = cache.borrow_mut();
        let cache = &mut cache.dfa;
        let mut dfa = Fsm {
            prog: prog,
            start: 0, // filled in below
            at: at,
            quit_after_match: false,
            last_match_si: STATE_UNKNOWN,
            last_cache_flush: at,
            cache: &mut cache.inner,
        };
        let (empty_flags, state_flags) = dfa.start_flags(text, at);
        dfa.start =
            match dfa.start_state(&mut cache.qcur, empty_flags, state_flags) {
                None => return Result::Quit,
                Some(STATE_DEAD) => return Result::NoMatch(at),
                Some(si) => si,
            };
        debug_assert!(dfa.start != STATE_UNKNOWN);
        let result = dfa.exec_at(&mut cache.qcur, &mut cache.qnext, text);
        if result.is_match() {
            if matches.len() == 1 {
                matches[0] = true;
            } else {
                debug_assert!(dfa.last_match_si != STATE_UNKNOWN);
                debug_assert!(dfa.last_match_si != STATE_DEAD);
                for ip in dfa.state(dfa.last_match_si).inst_ptrs() {
                    if let Inst::Match(slot) = dfa.prog[ip] {
                        matches[slot] = true;
                    }
                }
            }
        }
        result
    }

    /// Executes the DFA on a forward NFA.
    ///
    /// {qcur,qnext} are scratch ordered sets which may be non-empty.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn exec_at(
        &mut self,
        qcur: &mut SparseSet,
        qnext: &mut SparseSet,
        text: &[u8],
    ) -> Result<usize> {
        // For the most part, the DFA is basically:
        //
        //   last_match = null
        //   while current_byte != EOF:
        //     si = current_state.next[current_byte]
        //     if si is match
        //       last_match = si
        //   return last_match
        //
        // However, we need to deal with a few things:
        //
        //   1. This is an *online* DFA, so the current state's next list
        //      may not point to anywhere yet, so we must go out and compute
        //      them. (They are then cached into the current state's next list
        //      to avoid re-computation.)
        //   2. If we come across a state that is known to be dead (i.e., never
        //      leads to a match), then we can quit early.
        //   3. If the caller just wants to know if a match occurs, then we
        //      can quit as soon as we know we have a match. (Full leftmost
        //      first semantics require continuing on.)
        //   4. If we're in the start state, then we can use a pre-computed set
        //      of prefix literals to skip quickly along the input.
        //   5. After the input is exhausted, we run the DFA on one symbol
        //      that stands for EOF. This is useful for handling empty width
        //      assertions.
        //   6. We can't actually do state.next[byte]. Instead, we have to do
        //      state.next[byte_classes[byte]], which permits us to keep the
        //      'next' list very small.
        //
        // Since there's a bunch of extra stuff we need to consider, we do some
        // pretty hairy tricks to get the inner loop to run as fast as
        // possible.
        debug_assert!(!self.prog.is_reverse);

        // The last match is the currently known ending match position. It is
        // reported as an index to the most recent byte that resulted in a
        // transition to a match state and is always stored in capture slot `1`
        // when searching forwards. Its maximum value is `text.len()`.
        let mut result = Result::NoMatch(self.at);
        let (mut prev_si, mut next_si) = (self.start, self.start);
        let mut at = self.at;
        while at < text.len() {
            // This is the real inner loop. We take advantage of special bits
            // set in the state pointer to determine whether a state is in the
            // "common" case or not. Specifically, the common case is a
            // non-match non-start non-dead state that has already been
            // computed. So long as we remain in the common case, this inner
            // loop will chew through the input.
            //
            // We also unroll the loop 4 times to amortize the cost of checking
            // whether we've consumed the entire input. We are also careful
            // to make sure that `prev_si` always represents the previous state
            // and `next_si` always represents the next state after the loop
            // exits, even if it isn't always true inside the loop.
            while next_si <= STATE_MAX && at < text.len() {
                // Argument for safety is in the definition of next_si.
                prev_si = unsafe { self.next_si(next_si, text, at) };
                at += 1;
                if prev_si > STATE_MAX || at + 2 >= text.len() {
                    mem::swap(&mut prev_si, &mut next_si);
                    break;
                }
                next_si = unsafe { self.next_si(prev_si, text, at) };
                at += 1;
                if next_si > STATE_MAX {
                    break;
                }
                prev_si = unsafe { self.next_si(next_si, text, at) };
                at += 1;
                if prev_si > STATE_MAX {
                    mem::swap(&mut prev_si, &mut next_si);
                    break;
                }
                next_si = unsafe { self.next_si(prev_si, text, at) };
                at += 1;
            }
            if next_si & STATE_MATCH > 0 {
                // A match state is outside of the common case because it needs
                // special case analysis. In particular, we need to record the
                // last position as having matched and possibly quit the DFA if
                // we don't need to keep matching.
                next_si &= !STATE_MATCH;
                result = Result::Match(at - 1);
                if self.quit_after_match {
                    return result;
                }
                self.last_match_si = next_si;
                prev_si = next_si;

                // This permits short-circuiting when matching a regex set.
                // In particular, if this DFA state contains only match states,
                // then it's impossible to extend the set of matches since
                // match states are final. Therefore, we can quit.
                if self.prog.matches.len() > 1 {
                    let state = self.state(next_si);
                    let just_matches =
                        state.inst_ptrs().all(|ip| self.prog[ip].is_match());
                    if just_matches {
                        return result;
                    }
                }

                // Another inner loop! If the DFA stays in this particular
                // match state, then we can rip through all of the input
                // very quickly, and only recording the match location once
                // we've left this particular state.
                let cur = at;
                while (next_si & !STATE_MATCH) == prev_si
                    && at + 2 < text.len()
                {
                    // Argument for safety is in the definition of next_si.
                    next_si = unsafe {
                        self.next_si(next_si & !STATE_MATCH, text, at)
                    };
                    at += 1;
                }
                if at > cur {
                    result = Result::Match(at - 2);
                }
            } else if next_si & STATE_START > 0 {
                // A start state isn't in the common case because we may
                // want to do quick prefix scanning. If the program doesn't
                // have a detected prefix, then start states are actually
                // considered common and this case is never reached.
                debug_assert!(self.has_prefix());
                next_si &= !STATE_START;
                prev_si = next_si;
                at = match self.prefix_at(text, at) {
                    None => return Result::NoMatch(text.len()),
                    Some(i) => i,
                };
            } else if next_si >= STATE_UNKNOWN {
                if next_si == STATE_QUIT {
                    return Result::Quit;
                }
                // Finally, this corresponds to the case where the transition
                // entered a state that can never lead to a match or a state
                // that hasn't been computed yet. The latter being the "slow"
                // path.
                let byte = Byte::byte(text[at - 1]);
                // We no longer care about the special bits in the state
                // pointer.
                prev_si &= STATE_MAX;
                // Record where we are. This is used to track progress for
                // determining whether we should quit if we've flushed the
                // cache too much.
                self.at = at;
                next_si = match self.next_state(qcur, qnext, prev_si, byte) {
                    None => return Result::Quit,
                    Some(STATE_DEAD) => return result.set_non_match(at),
                    Some(si) => si,
                };
                debug_assert!(next_si != STATE_UNKNOWN);
                if next_si & STATE_MATCH > 0 {
                    next_si &= !STATE_MATCH;
                    result = Result::Match(at - 1);
                    if self.quit_after_match {
                        return result;
                    }
                    self.last_match_si = next_si;
                }
                prev_si = next_si;
            } else {
                prev_si = next_si;
            }
        }

        // Run the DFA once more on the special EOF sentinel value.
        // We don't care about the special bits in the state pointer any more,
        // so get rid of them.
        prev_si &= STATE_MAX;
        prev_si = match self.next_state(qcur, qnext, prev_si, Byte::eof()) {
            None => return Result::Quit,
            Some(STATE_DEAD) => return result.set_non_match(text.len()),
            Some(si) => si & !STATE_START,
        };
        debug_assert!(prev_si != STATE_UNKNOWN);
        if prev_si & STATE_MATCH > 0 {
            prev_si &= !STATE_MATCH;
            self.last_match_si = prev_si;
            result = Result::Match(text.len());
        }
        result
    }

    /// Executes the DFA on a reverse NFA.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn exec_at_reverse(
        &mut self,
        qcur: &mut SparseSet,
        qnext: &mut SparseSet,
        text: &[u8],
    ) -> Result<usize> {
        // The comments in `exec_at` above mostly apply here too. The main
        // difference is that we move backwards over the input and we look for
        // the longest possible match instead of the leftmost-first match.
        //
        // N.B. The code duplication here is regrettable. Efforts to improve
        // it without sacrificing performance are welcome. ---AG
        debug_assert!(self.prog.is_reverse);
        let mut result = Result::NoMatch(self.at);
        let (mut prev_si, mut next_si) = (self.start, self.start);
        let mut at = self.at;
        while at > 0 {
            while next_si <= STATE_MAX && at > 0 {
                // Argument for safety is in the definition of next_si.
                at -= 1;
                prev_si = unsafe { self.next_si(next_si, text, at) };
                if prev_si > STATE_MAX || at <= 4 {
                    mem::swap(&mut prev_si, &mut next_si);
                    break;
                }
                at -= 1;
                next_si = unsafe { self.next_si(prev_si, text, at) };
                if next_si > STATE_MAX {
                    break;
                }
                at -= 1;
                prev_si = unsafe { self.next_si(next_si, text, at) };
                if prev_si > STATE_MAX {
                    mem::swap(&mut prev_si, &mut next_si);
                    break;
                }
                at -= 1;
                next_si = unsafe { self.next_si(prev_si, text, at) };
            }
            if next_si & STATE_MATCH > 0 {
                next_si &= !STATE_MATCH;
                result = Result::Match(at + 1);
                if self.quit_after_match {
                    return result;
                }
                self.last_match_si = next_si;
                prev_si = next_si;
                let cur = at;
                while (next_si & !STATE_MATCH) == prev_si && at >= 2 {
                    // Argument for safety is in the definition of next_si.
                    at -= 1;
                    next_si = unsafe {
                        self.next_si(next_si & !STATE_MATCH, text, at)
                    };
                }
                if at < cur {
                    result = Result::Match(at + 2);
                }
            } else if next_si >= STATE_UNKNOWN {
                if next_si == STATE_QUIT {
                    return Result::Quit;
                }
                let byte = Byte::byte(text[at]);
                prev_si &= STATE_MAX;
                self.at = at;
                next_si = match self.next_state(qcur, qnext, prev_si, byte) {
                    None => return Result::Quit,
                    Some(STATE_DEAD) => return result.set_non_match(at),
                    Some(si) => si,
                };
                debug_assert!(next_si != STATE_UNKNOWN);
                if next_si & STATE_MATCH > 0 {
                    next_si &= !STATE_MATCH;
                    result = Result::Match(at + 1);
                    if self.quit_after_match {
                        return result;
                    }
                    self.last_match_si = next_si;
                }
                prev_si = next_si;
            } else {
                prev_si = next_si;
            }
        }

        // Run the DFA once more on the special EOF sentinel value.
        prev_si = match self.next_state(qcur, qnext, prev_si, Byte::eof()) {
            None => return Result::Quit,
            Some(STATE_DEAD) => return result.set_non_match(0),
            Some(si) => si,
        };
        debug_assert!(prev_si != STATE_UNKNOWN);
        if prev_si & STATE_MATCH > 0 {
            prev_si &= !STATE_MATCH;
            self.last_match_si = prev_si;
            result = Result::Match(0);
        }
        result
    }

    /// next_si transitions to the next state, where the transition input
    /// corresponds to text[i].
    ///
    /// This elides bounds checks, and is therefore unsafe.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    unsafe fn next_si(&self, si: StatePtr, text: &[u8], i: usize) -> StatePtr {
        // What is the argument for safety here?
        // We have three unchecked accesses that could possibly violate safety:
        //
        //   1. The given byte of input (`text[i]`).
        //   2. The class of the byte of input (`classes[text[i]]`).
        //   3. The transition for the class (`trans[si + cls]`).
        //
        // (1) is only safe when calling next_si is guarded by
        // `i < text.len()`.
        //
        // (2) is the easiest case to guarantee since `text[i]` is always a
        // `u8` and `self.prog.byte_classes` always has length `u8::MAX`.
        // (See `ByteClassSet.byte_classes` in `compile.rs`.)
        //
        // (3) is only safe if (1)+(2) are safe. Namely, the transitions
        // of every state are defined to have length equal to the number of
        // byte classes in the program. Therefore, a valid class leads to a
        // valid transition. (All possible transitions are valid lookups, even
        // if it points to a state that hasn't been computed yet.) (3) also
        // relies on `si` being correct, but StatePtrs should only ever be
        // retrieved from the transition table, which ensures they are correct.
        debug_assert!(i < text.len());
        let b = *text.get_unchecked(i);
        debug_assert!((b as usize) < self.prog.byte_classes.len());
        let cls = *self.prog.byte_classes.get_unchecked(b as usize);
        self.cache.trans.next_unchecked(si, cls as usize)
    }

    /// Computes the next state given the current state and the current input
    /// byte (which may be EOF).
    ///
    /// If STATE_DEAD is returned, then there is no valid state transition.
    /// This implies that no permutation of future input can lead to a match
    /// state.
    ///
    /// STATE_UNKNOWN can never be returned.
    fn exec_byte(
        &mut self,
        qcur: &mut SparseSet,
        qnext: &mut SparseSet,
        mut si: StatePtr,
        b: Byte,
    ) -> Option<StatePtr> {
        use prog::Inst::*;

        // Initialize a queue with the current DFA state's NFA states.
        qcur.clear();
        for ip in self.state(si).inst_ptrs() {
            qcur.insert(ip);
        }

        // Before inspecting the current byte, we may need to also inspect
        // whether the position immediately preceding the current byte
        // satisfies the empty assertions found in the current state.
        //
        // We only need to do this step if there are any empty assertions in
        // the current state.
        let is_word_last = self.state(si).flags().is_word();
        let is_word = b.is_ascii_word();
        if self.state(si).flags().has_empty() {
            // Compute the flags immediately preceding the current byte.
            // This means we only care about the "end" or "end line" flags.
            // (The "start" flags are computed immediately following the
            // current byte and are handled below.)
            let mut flags = EmptyFlags::default();
            if b.is_eof() {
                flags.end = true;
                flags.end_line = true;
            } else if b.as_byte().map_or(false, |b| b == b'\n') {
                flags.end_line = true;
            }
            if is_word_last == is_word {
                flags.not_word_boundary = true;
            } else {
                flags.word_boundary = true;
            }
            // Now follow epsilon transitions from every NFA state, but make
            // sure we only follow transitions that satisfy our flags.
            qnext.clear();
            for &ip in &*qcur {
                self.follow_epsilons(usize_to_u32(ip), qnext, flags);
            }
            mem::swap(qcur, qnext);
        }

        // Now we set flags for immediately after the current byte. Since start
        // states are processed separately, and are the only states that can
        // have the StartText flag set, we therefore only need to worry about
        // the StartLine flag here.
        //
        // We do also keep track of whether this DFA state contains a NFA state
        // that is a matching state. This is precisely how we delay the DFA
        // matching by one byte in order to process the special EOF sentinel
        // byte. Namely, if this DFA state containing a matching NFA state,
        // then it is the *next* DFA state that is marked as a match.
        let mut empty_flags = EmptyFlags::default();
        let mut state_flags = StateFlags::default();
        empty_flags.start_line = b.as_byte().map_or(false, |b| b == b'\n');
        if b.is_ascii_word() {
            state_flags.set_word();
        }
        // Now follow all epsilon transitions again, but only after consuming
        // the current byte.
        qnext.clear();
        for &ip in &*qcur {
            match self.prog[ip as usize] {
                // These states never happen in a byte-based program.
                Char(_) | Ranges(_) => unreachable!(),
                // These states are handled when following epsilon transitions.
                Save(_) | Split(_) | EmptyLook(_) => {}
                Match(_) => {
                    state_flags.set_match();
                    if !self.continue_past_first_match() {
                        break;
                    } else if self.prog.matches.len() > 1
                        && !qnext.contains(ip as usize)
                    {
                        // If we are continuing on to find other matches,
                        // then keep a record of the match states we've seen.
                        qnext.insert(ip);
                    }
                }
                Bytes(ref inst) => {
                    if b.as_byte().map_or(false, |b| inst.matches(b)) {
                        self.follow_epsilons(
                            inst.goto as InstPtr,
                            qnext,
                            empty_flags,
                        );
                    }
                }
            }
        }

        let cache = if b.is_eof() && self.prog.matches.len() > 1 {
            // If we're processing the last byte of the input and we're
            // matching a regex set, then make the next state contain the
            // previous states transitions. We do this so that the main
            // matching loop can extract all of the match instructions.
            mem::swap(qcur, qnext);
            // And don't cache this state because it's totally bunk.
            false
        } else {
            true
        };

        // We've now built up the set of NFA states that ought to comprise the
        // next DFA state, so try to find it in the cache, and if it doesn't
        // exist, cache it.
        //
        // N.B. We pass `&mut si` here because the cache may clear itself if
        // it has gotten too full. When that happens, the location of the
        // current state may change.
        let mut next =
            match self.cached_state(qnext, state_flags, Some(&mut si)) {
                None => return None,
                Some(next) => next,
            };
        if (self.start & !STATE_START) == next {
            // Start states can never be match states since all matches are
            // delayed by one byte.
            debug_assert!(!self.state(next).flags().is_match());
            next = self.start_ptr(next);
        }
        if next <= STATE_MAX && self.state(next).flags().is_match() {
            next |= STATE_MATCH;
        }
        debug_assert!(next != STATE_UNKNOWN);
        // And now store our state in the current state's next list.
        if cache {
            let cls = self.byte_class(b);
            self.cache.trans.set_next(si, cls, next);
        }
        Some(next)
    }

    /// Follows the epsilon transitions starting at (and including) `ip`. The
    /// resulting states are inserted into the ordered set `q`.
    ///
    /// Conditional epsilon transitions (i.e., empty width assertions) are only
    /// followed if they are satisfied by the given flags, which should
    /// represent the flags set at the current location in the input.
    ///
    /// If the current location corresponds to the empty string, then only the
    /// end line and/or end text flags may be set. If the current location
    /// corresponds to a real byte in the input, then only the start line
    /// and/or start text flags may be set.
    ///
    /// As an exception to the above, when finding the initial state, any of
    /// the above flags may be set:
    ///
    /// If matching starts at the beginning of the input, then start text and
    /// start line should be set. If the input is empty, then end text and end
    /// line should also be set.
    ///
    /// If matching starts after the beginning of the input, then only start
    /// line should be set if the preceding byte is `\n`. End line should never
    /// be set in this case. (Even if the following byte is a `\n`, it will
    /// be handled in a subsequent DFA state.)
    fn follow_epsilons(
        &mut self,
        ip: InstPtr,
        q: &mut SparseSet,
        flags: EmptyFlags,
    ) {
        use prog::EmptyLook::*;
        use prog::Inst::*;

        // We need to traverse the NFA to follow epsilon transitions, so avoid
        // recursion with an explicit stack.
        self.cache.stack.push(ip);
        while let Some(mut ip) = self.cache.stack.pop() {
            // Try to munch through as many states as possible without
            // pushes/pops to the stack.
            loop {
                // Don't visit states we've already added.
                if q.contains(ip as usize) {
                    break;
                }
                q.insert(ip as usize);
                match self.prog[ip as usize] {
                    Char(_) | Ranges(_) => unreachable!(),
                    Match(_) | Bytes(_) => {
                        break;
                    }
                    EmptyLook(ref inst) => {
                        // Only follow empty assertion states if our flags
                        // satisfy the assertion.
                        match inst.look {
                            StartLine if flags.start_line => {
                                ip = inst.goto as InstPtr;
                            }
                            EndLine if flags.end_line => {
                                ip = inst.goto as InstPtr;
                            }
                            StartText if flags.start => {
                                ip = inst.goto as InstPtr;
                            }
                            EndText if flags.end => {
                                ip = inst.goto as InstPtr;
                            }
                            WordBoundaryAscii if flags.word_boundary => {
                                ip = inst.goto as InstPtr;
                            }
                            NotWordBoundaryAscii
                                if flags.not_word_boundary =>
                            {
                                ip = inst.goto as InstPtr;
                            }
                            WordBoundary if flags.word_boundary => {
                                ip = inst.goto as InstPtr;
                            }
                            NotWordBoundary if flags.not_word_boundary => {
                                ip = inst.goto as InstPtr;
                            }
                            StartLine | EndLine | StartText | EndText
                            | WordBoundaryAscii | NotWordBoundaryAscii
                            | WordBoundary | NotWordBoundary => {
                                break;
                            }
                        }
                    }
                    Save(ref inst) => {
                        ip = inst.goto as InstPtr;
                    }
                    Split(ref inst) => {
                        self.cache.stack.push(inst.goto2 as InstPtr);
                        ip = inst.goto1 as InstPtr;
                    }
                }
            }
        }
    }

    /// Find a previously computed state matching the given set of instructions
    /// and is_match bool.
    ///
    /// The given set of instructions should represent a single state in the
    /// NFA along with all states reachable without consuming any input.
    ///
    /// The is_match bool should be true if and only if the preceding DFA state
    /// contains an NFA matching state. The cached state produced here will
    /// then signify a match. (This enables us to delay a match by one byte,
    /// in order to account for the EOF sentinel byte.)
    ///
    /// If the cache is full, then it is wiped before caching a new state.
    ///
    /// The current state should be specified if it exists, since it will need
    /// to be preserved if the cache clears itself. (Start states are
    /// always saved, so they should not be passed here.) It takes a mutable
    /// pointer to the index because if the cache is cleared, the state's
    /// location may change.
    fn cached_state(
        &mut self,
        q: &SparseSet,
        mut state_flags: StateFlags,
        current_state: Option<&mut StatePtr>,
    ) -> Option<StatePtr> {
        // If we couldn't come up with a non-empty key to represent this state,
        // then it is dead and can never lead to a match.
        //
        // Note that inst_flags represent the set of empty width assertions
        // in q. We use this as an optimization in exec_byte to determine when
        // we should follow epsilon transitions at the empty string preceding
        // the current byte.
        let key = match self.cached_state_key(q, &mut state_flags) {
            None => return Some(STATE_DEAD),
            Some(v) => v,
        };
        // In the cache? Cool. Done.
        if let Some(si) = self.cache.compiled.get_ptr(&key) {
            return Some(si);
        }
        // If the cache has gotten too big, wipe it.
        if self.approximate_size() > self.prog.dfa_size_limit
            && !self.clear_cache_and_save(current_state)
        {
            // Ooops. DFA is giving up.
            return None;
        }
        // Allocate room for our state and add it.
        self.add_state(key)
    }

    /// Produces a key suitable for describing a state in the DFA cache.
    ///
    /// The key invariant here is that equivalent keys are produced for any two
    /// sets of ordered NFA states (and toggling of whether the previous NFA
    /// states contain a match state) that do not discriminate a match for any
    /// input.
    ///
    /// Specifically, q should be an ordered set of NFA states and is_match
    /// should be true if and only if the previous NFA states contained a match
    /// state.
    fn cached_state_key(
        &mut self,
        q: &SparseSet,
        state_flags: &mut StateFlags,
    ) -> Option<State> {
        use prog::Inst::*;

        // We need to build up enough information to recognize pre-built states
        // in the DFA. Generally speaking, this includes every instruction
        // except for those which are purely epsilon transitions, e.g., the
        // Save and Split instructions.
        //
        // Empty width assertions are also epsilon transitions, but since they
        // are conditional, we need to make them part of a state's key in the
        // cache.

        let mut insts =
            mem::replace(&mut self.cache.insts_scratch_space, vec![]);
        insts.clear();
        // Reserve 1 byte for flags.
        insts.push(0);

        let mut prev = 0;
        for &ip in q {
            let ip = usize_to_u32(ip);
            match self.prog[ip as usize] {
                Char(_) | Ranges(_) => unreachable!(),
                Save(_) | Split(_) => {}
                Bytes(_) => push_inst_ptr(&mut insts, &mut prev, ip),
                EmptyLook(_) => {
                    state_flags.set_empty();
                    push_inst_ptr(&mut insts, &mut prev, ip)
                }
                Match(_) => {
                    push_inst_ptr(&mut insts, &mut prev, ip);
                    if !self.continue_past_first_match() {
                        break;
                    }
                }
            }
        }
        // If we couldn't transition to any other instructions and we didn't
        // see a match when expanding NFA states previously, then this is a
        // dead state and no amount of additional input can transition out
        // of this state.
        let opt_state = if insts.len() == 1 && !state_flags.is_match() {
            None
        } else {
            let StateFlags(f) = *state_flags;
            insts[0] = f;
            Some(State { data: Arc::from(&*insts) })
        };
        self.cache.insts_scratch_space = insts;
        opt_state
    }

    /// Clears the cache, but saves and restores current_state if it is not
    /// none.
    ///
    /// The current state must be provided here in case its location in the
    /// cache changes.
    ///
    /// This returns false if the cache is not cleared and the DFA should
    /// give up.
    fn clear_cache_and_save(
        &mut self,
        current_state: Option<&mut StatePtr>,
    ) -> bool {
        if self.cache.compiled.is_empty() {
            // Nothing to clear...
            return true;
        }
        match current_state {
            None => self.clear_cache(),
            Some(si) => {
                let cur = self.state(*si).clone();
                if !self.clear_cache() {
                    return false;
                }
                // The unwrap is OK because we just cleared the cache and
                // therefore know that the next state pointer won't exceed
                // STATE_MAX.
                *si = self.restore_state(cur).unwrap();
                true
            }
        }
    }

    /// Wipes the state cache, but saves and restores the current start state.
    ///
    /// This returns false if the cache is not cleared and the DFA should
    /// give up.
    fn clear_cache(&mut self) -> bool {
        // Bail out of the DFA if we're moving too "slowly."
        // A heuristic from RE2: assume the DFA is too slow if it is processing
        // 10 or fewer bytes per state.
        // Additionally, we permit the cache to be flushed a few times before
        // caling it quits.
        let nstates = self.cache.compiled.len();
        if self.cache.flush_count >= 3
            && self.at >= self.last_cache_flush
            && (self.at - self.last_cache_flush) <= 10 * nstates
        {
            return false;
        }
        // Update statistics tracking cache flushes.
        self.last_cache_flush = self.at;
        self.cache.flush_count += 1;

        // OK, actually flush the cache.
        let start = self.state(self.start & !STATE_START).clone();
        let last_match = if self.last_match_si <= STATE_MAX {
            Some(self.state(self.last_match_si).clone())
        } else {
            None
        };
        self.cache.reset_size();
        self.cache.trans.clear();
        self.cache.compiled.clear();
        for s in &mut self.cache.start_states {
            *s = STATE_UNKNOWN;
        }
        // The unwraps are OK because we just cleared the cache and therefore
        // know that the next state pointer won't exceed STATE_MAX.
        let start_ptr = self.restore_state(start).unwrap();
        self.start = self.start_ptr(start_ptr);
        if let Some(last_match) = last_match {
            self.last_match_si = self.restore_state(last_match).unwrap();
        }
        true
    }

    /// Restores the given state back into the cache, and returns a pointer
    /// to it.
    fn restore_state(&mut self, state: State) -> Option<StatePtr> {
        // If we've already stored this state, just return a pointer to it.
        // None will be the wiser.
        if let Some(si) = self.cache.compiled.get_ptr(&state) {
            return Some(si);
        }
        self.add_state(state)
    }

    /// Returns the next state given the current state si and current byte
    /// b. {qcur,qnext} are used as scratch space for storing ordered NFA
    /// states.
    ///
    /// This tries to fetch the next state from the cache, but if that fails,
    /// it computes the next state, caches it and returns a pointer to it.
    ///
    /// The pointer can be to a real state, or it can be STATE_DEAD.
    /// STATE_UNKNOWN cannot be returned.
    ///
    /// None is returned if a new state could not be allocated (i.e., the DFA
    /// ran out of space and thinks it's running too slowly).
    fn next_state(
        &mut self,
        qcur: &mut SparseSet,
        qnext: &mut SparseSet,
        si: StatePtr,
        b: Byte,
    ) -> Option<StatePtr> {
        if si == STATE_DEAD {
            return Some(STATE_DEAD);
        }
        match self.cache.trans.next(si, self.byte_class(b)) {
            STATE_UNKNOWN => self.exec_byte(qcur, qnext, si, b),
            STATE_QUIT => None,
            STATE_DEAD => Some(STATE_DEAD),
            nsi => Some(nsi),
        }
    }

    /// Computes and returns the start state, where searching begins at
    /// position `at` in `text`. If the state has already been computed,
    /// then it is pulled from the cache. If the state hasn't been cached,
    /// then it is computed, cached and a pointer to it is returned.
    ///
    /// This may return STATE_DEAD but never STATE_UNKNOWN.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn start_state(
        &mut self,
        q: &mut SparseSet,
        empty_flags: EmptyFlags,
        state_flags: StateFlags,
    ) -> Option<StatePtr> {
        // Compute an index into our cache of start states based on the set
        // of empty/state flags set at the current position in the input. We
        // don't use every flag since not all flags matter. For example, since
        // matches are delayed by one byte, start states can never be match
        // states.
        let flagi = {
            (((empty_flags.start as u8) << 0)
                | ((empty_flags.end as u8) << 1)
                | ((empty_flags.start_line as u8) << 2)
                | ((empty_flags.end_line as u8) << 3)
                | ((empty_flags.word_boundary as u8) << 4)
                | ((empty_flags.not_word_boundary as u8) << 5)
                | ((state_flags.is_word() as u8) << 6)) as usize
        };
        match self.cache.start_states[flagi] {
            STATE_UNKNOWN => {}
            STATE_DEAD => return Some(STATE_DEAD),
            si => return Some(si),
        }
        q.clear();
        let start = usize_to_u32(self.prog.start);
        self.follow_epsilons(start, q, empty_flags);
        // Start states can never be match states because we delay every match
        // by one byte. Given an empty string and an empty match, the match
        // won't actually occur until the DFA processes the special EOF
        // sentinel byte.
        let sp = match self.cached_state(q, state_flags, None) {
            None => return None,
            Some(sp) => self.start_ptr(sp),
        };
        self.cache.start_states[flagi] = sp;
        Some(sp)
    }

    /// Computes the set of starting flags for the given position in text.
    ///
    /// This should only be used when executing the DFA forwards over the
    /// input.
    fn start_flags(&self, text: &[u8], at: usize) -> (EmptyFlags, StateFlags) {
        let mut empty_flags = EmptyFlags::default();
        let mut state_flags = StateFlags::default();
        empty_flags.start = at == 0;
        empty_flags.end = text.is_empty();
        empty_flags.start_line = at == 0 || text[at - 1] == b'\n';
        empty_flags.end_line = text.is_empty();

        let is_word_last = at > 0 && Byte::byte(text[at - 1]).is_ascii_word();
        let is_word = at < text.len() && Byte::byte(text[at]).is_ascii_word();
        if is_word_last {
            state_flags.set_word();
        }
        if is_word == is_word_last {
            empty_flags.not_word_boundary = true;
        } else {
            empty_flags.word_boundary = true;
        }
        (empty_flags, state_flags)
    }

    /// Computes the set of starting flags for the given position in text.
    ///
    /// This should only be used when executing the DFA in reverse over the
    /// input.
    fn start_flags_reverse(
        &self,
        text: &[u8],
        at: usize,
    ) -> (EmptyFlags, StateFlags) {
        let mut empty_flags = EmptyFlags::default();
        let mut state_flags = StateFlags::default();
        empty_flags.start = at == text.len();
        empty_flags.end = text.is_empty();
        empty_flags.start_line = at == text.len() || text[at] == b'\n';
        empty_flags.end_line = text.is_empty();

        let is_word_last =
            at < text.len() && Byte::byte(text[at]).is_ascii_word();
        let is_word = at > 0 && Byte::byte(text[at - 1]).is_ascii_word();
        if is_word_last {
            state_flags.set_word();
        }
        if is_word == is_word_last {
            empty_flags.not_word_boundary = true;
        } else {
            empty_flags.word_boundary = true;
        }
        (empty_flags, state_flags)
    }

    /// Returns a reference to a State given a pointer to it.
    fn state(&self, si: StatePtr) -> &State {
        self.cache.compiled.get_state(si).unwrap()
    }

    /// Adds the given state to the DFA.
    ///
    /// This allocates room for transitions out of this state in
    /// self.cache.trans. The transitions can be set with the returned
    /// StatePtr.
    ///
    /// If None is returned, then the state limit was reached and the DFA
    /// should quit.
    fn add_state(&mut self, state: State) -> Option<StatePtr> {
        // This will fail if the next state pointer exceeds STATE_PTR. In
        // practice, the cache limit will prevent us from ever getting here,
        // but maybe callers will set the cache size to something ridiculous...
        let si = match self.cache.trans.add() {
            None => return None,
            Some(si) => si,
        };
        // If the program has a Unicode word boundary, then set any transitions
        // for non-ASCII bytes to STATE_QUIT. If the DFA stumbles over such a
        // transition, then it will quit and an alternative matching engine
        // will take over.
        if self.prog.has_unicode_word_boundary {
            for b in 128..256 {
                let cls = self.byte_class(Byte::byte(b as u8));
                self.cache.trans.set_next(si, cls, STATE_QUIT);
            }
        }
        // Finally, put our actual state on to our heap of states and index it
        // so we can find it later.
        self.cache.size += self.cache.trans.state_heap_size()
            + state.data.len()
            + (2 * mem::size_of::<State>())
            + mem::size_of::<StatePtr>();
        self.cache.compiled.insert(state, si);
        // Transition table and set of states and map should all be in sync.
        debug_assert!(
            self.cache.compiled.len() == self.cache.trans.num_states()
        );
        Some(si)
    }

    /// Quickly finds the next occurrence of any literal prefixes in the regex.
    /// If there are no literal prefixes, then the current position is
    /// returned. If there are literal prefixes and one could not be found,
    /// then None is returned.
    ///
    /// This should only be called when the DFA is in a start state.
    fn prefix_at(&self, text: &[u8], at: usize) -> Option<usize> {
        self.prog.prefixes.find(&text[at..]).map(|(s, _)| at + s)
    }

    /// Returns the number of byte classes required to discriminate transitions
    /// in each state.
    ///
    /// invariant: num_byte_classes() == len(State.next)
    fn num_byte_classes(&self) -> usize {
        // We add 1 to account for the special EOF byte.
        (self.prog.byte_classes[255] as usize + 1) + 1
    }

    /// Given an input byte or the special EOF sentinel, return its
    /// corresponding byte class.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn byte_class(&self, b: Byte) -> usize {
        match b.as_byte() {
            None => self.num_byte_classes() - 1,
            Some(b) => self.u8_class(b),
        }
    }

    /// Like byte_class, but explicitly for u8s.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn u8_class(&self, b: u8) -> usize {
        self.prog.byte_classes[b as usize] as usize
    }

    /// Returns true if the DFA should continue searching past the first match.
    ///
    /// Leftmost first semantics in the DFA are preserved by not following NFA
    /// transitions after the first match is seen.
    ///
    /// On occasion, we want to avoid leftmost first semantics to find either
    /// the longest match (for reverse search) or all possible matches (for
    /// regex sets).
    fn continue_past_first_match(&self) -> bool {
        self.prog.is_reverse || self.prog.matches.len() > 1
    }

    /// Returns true if there is a prefix we can quickly search for.
    fn has_prefix(&self) -> bool {
        !self.prog.is_reverse
            && !self.prog.prefixes.is_empty()
            && !self.prog.is_anchored_start
    }

    /// Sets the STATE_START bit in the given state pointer if and only if
    /// we have a prefix to scan for.
    ///
    /// If there's no prefix, then it's a waste to treat the start state
    /// specially.
    fn start_ptr(&self, si: StatePtr) -> StatePtr {
        if self.has_prefix() {
            si | STATE_START
        } else {
            si
        }
    }

    /// Approximate size returns the approximate heap space currently used by
    /// the DFA. It is used to determine whether the DFA's state cache needs to
    /// be wiped. Namely, it is possible that for certain regexes on certain
    /// inputs, a new state could be created for every byte of input. (This is
    /// bad for memory use, so we bound it with a cache.)
    fn approximate_size(&self) -> usize {
        self.cache.size + self.prog.approximate_size()
    }
}

/// An abstraction for representing a map of states. The map supports two
/// different ways of state lookup. One is fast constant time access via a
/// state pointer. The other is a hashmap lookup based on the DFA's
/// constituent NFA states.
///
/// A DFA state internally uses an Arc such that we only need to store the
/// set of NFA states on the heap once, even though we support looking up
/// states by two different means. A more natural way to express this might
/// use raw pointers, but an Arc is safe and effectively achieves the same
/// thing.
#[derive(Debug)]
struct StateMap {
    /// The keys are not actually static but rely on always pointing to a
    /// buffer in `states` which will never be moved except when clearing
    /// the map or on drop, in which case the keys of this map will be
    /// removed before
    map: HashMap<State, StatePtr>,
    /// Our set of states. Note that `StatePtr / num_byte_classes` indexes
    /// this Vec rather than just a `StatePtr`.
    states: Vec<State>,
    /// The number of byte classes in the DFA. Used to index `states`.
    num_byte_classes: usize,
}

impl StateMap {
    fn new(num_byte_classes: usize) -> StateMap {
        StateMap {
            map: HashMap::new(),
            states: vec![],
            num_byte_classes: num_byte_classes,
        }
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn get_ptr(&self, state: &State) -> Option<StatePtr> {
        self.map.get(state).cloned()
    }

    fn get_state(&self, si: StatePtr) -> Option<&State> {
        self.states.get(si as usize / self.num_byte_classes)
    }

    fn insert(&mut self, state: State, si: StatePtr) {
        self.map.insert(state.clone(), si);
        self.states.push(state);
    }

    fn clear(&mut self) {
        self.map.clear();
        self.states.clear();
    }
}

impl Transitions {
    /// Create a new transition table.
    ///
    /// The number of byte classes corresponds to the stride. Every state will
    /// have `num_byte_classes` slots for transitions.
    fn new(num_byte_classes: usize) -> Transitions {
        Transitions { table: vec![], num_byte_classes: num_byte_classes }
    }

    /// Returns the total number of states currently in this table.
    fn num_states(&self) -> usize {
        self.table.len() / self.num_byte_classes
    }

    /// Allocates room for one additional state and returns a pointer to it.
    ///
    /// If there's no more room, None is returned.
    fn add(&mut self) -> Option<StatePtr> {
        let si = self.table.len();
        if si > STATE_MAX as usize {
            return None;
        }
        self.table.extend(repeat(STATE_UNKNOWN).take(self.num_byte_classes));
        Some(usize_to_u32(si))
    }

    /// Clears the table of all states.
    fn clear(&mut self) {
        self.table.clear();
    }

    /// Sets the transition from (si, cls) to next.
    fn set_next(&mut self, si: StatePtr, cls: usize, next: StatePtr) {
        self.table[si as usize + cls] = next;
    }

    /// Returns the transition corresponding to (si, cls).
    fn next(&self, si: StatePtr, cls: usize) -> StatePtr {
        self.table[si as usize + cls]
    }

    /// The heap size, in bytes, of a single state in the transition table.
    fn state_heap_size(&self) -> usize {
        self.num_byte_classes * mem::size_of::<StatePtr>()
    }

    /// Like `next`, but uses unchecked access and is therefore unsafe.
    unsafe fn next_unchecked(&self, si: StatePtr, cls: usize) -> StatePtr {
        debug_assert!((si as usize) < self.table.len());
        debug_assert!(cls < self.num_byte_classes);
        *self.table.get_unchecked(si as usize + cls)
    }
}

impl StateFlags {
    fn is_match(&self) -> bool {
        self.0 & 0b0000000_1 > 0
    }

    fn set_match(&mut self) {
        self.0 |= 0b0000000_1;
    }

    fn is_word(&self) -> bool {
        self.0 & 0b000000_1_0 > 0
    }

    fn set_word(&mut self) {
        self.0 |= 0b000000_1_0;
    }

    fn has_empty(&self) -> bool {
        self.0 & 0b00000_1_00 > 0
    }

    fn set_empty(&mut self) {
        self.0 |= 0b00000_1_00;
    }
}

impl Byte {
    fn byte(b: u8) -> Self {
        Byte(b as u16)
    }
    fn eof() -> Self {
        Byte(256)
    }
    fn is_eof(&self) -> bool {
        self.0 == 256
    }

    fn is_ascii_word(&self) -> bool {
        let b = match self.as_byte() {
            None => return false,
            Some(b) => b,
        };
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' => true,
            _ => false,
        }
    }

    fn as_byte(&self) -> Option<u8> {
        if self.is_eof() {
            None
        } else {
            Some(self.0 as u8)
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ips: Vec<usize> = self.inst_ptrs().collect();
        f.debug_struct("State")
            .field("flags", &self.flags())
            .field("insts", &ips)
            .finish()
    }
}

impl fmt::Debug for Transitions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut fmtd = f.debug_map();
        for si in 0..self.num_states() {
            let s = si * self.num_byte_classes;
            let e = s + self.num_byte_classes;
            fmtd.entry(&si.to_string(), &TransitionsRow(&self.table[s..e]));
        }
        fmtd.finish()
    }
}

struct TransitionsRow<'a>(&'a [StatePtr]);

impl<'a> fmt::Debug for TransitionsRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut fmtd = f.debug_map();
        for (b, si) in self.0.iter().enumerate() {
            match *si {
                STATE_UNKNOWN => {}
                STATE_DEAD => {
                    fmtd.entry(&vb(b as usize), &"DEAD");
                }
                si => {
                    fmtd.entry(&vb(b as usize), &si.to_string());
                }
            }
        }
        fmtd.finish()
    }
}

impl fmt::Debug for StateFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StateFlags")
            .field("is_match", &self.is_match())
            .field("is_word", &self.is_word())
            .field("has_empty", &self.has_empty())
            .finish()
    }
}

/// Helper function for formatting a byte as a nice-to-read escaped string.
fn vb(b: usize) -> String {
    use std::ascii::escape_default;

    if b > ::std::u8::MAX as usize {
        "EOF".to_owned()
    } else {
        let escaped = escape_default(b as u8).collect::<Vec<u8>>();
        String::from_utf8_lossy(&escaped).into_owned()
    }
}

fn usize_to_u32(n: usize) -> u32 {
    if (n as u64) > (::std::u32::MAX as u64) {
        panic!("BUG: {} is too big to fit into u32", n)
    }
    n as u32
}

#[allow(dead_code)] // useful for debugging
fn show_state_ptr(si: StatePtr) -> String {
    let mut s = format!("{:?}", si & STATE_MAX);
    if si == STATE_UNKNOWN {
        s = format!("{} (unknown)", s);
    }
    if si == STATE_DEAD {
        s = format!("{} (dead)", s);
    }
    if si == STATE_QUIT {
        s = format!("{} (quit)", s);
    }
    if si & STATE_START > 0 {
        s = format!("{} (start)", s);
    }
    if si & STATE_MATCH > 0 {
        s = format!("{} (match)", s);
    }
    s
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn write_vari32(data: &mut Vec<u8>, n: i32) {
    let mut un = (n as u32) << 1;
    if n < 0 {
        un = !un;
    }
    write_varu32(data, un)
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn read_vari32(data: &[u8]) -> (i32, usize) {
    let (un, i) = read_varu32(data);
    let mut n = (un >> 1) as i32;
    if un & 1 != 0 {
        n = !n;
    }
    (n, i)
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn write_varu32(data: &mut Vec<u8>, mut n: u32) {
    while n >= 0b1000_0000 {
        data.push((n as u8) | 0b1000_0000);
        n >>= 7;
    }
    data.push(n as u8);
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn read_varu32(data: &[u8]) -> (u32, usize) {
    let mut n: u32 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in data.iter().enumerate() {
        if b < 0b1000_0000 {
            return (n | ((b as u32) << shift), i + 1);
        }
        n |= ((b as u32) & 0b0111_1111) << shift;
        shift += 7;
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use super::{
        push_inst_ptr, read_vari32, read_varu32, write_vari32, write_varu32,
        State, StateFlags,
    };
    use quickcheck::{quickcheck, QuickCheck, StdGen};
    use std::sync::Arc;

    #[test]
    fn prop_state_encode_decode() {
        fn p(ips: Vec<u32>, flags: u8) -> bool {
            let mut data = vec![flags];
            let mut prev = 0;
            for &ip in ips.iter() {
                push_inst_ptr(&mut data, &mut prev, ip);
            }
            let state = State { data: Arc::from(&data[..]) };

            let expected: Vec<usize> =
                ips.into_iter().map(|ip| ip as usize).collect();
            let got: Vec<usize> = state.inst_ptrs().collect();
            expected == got && state.flags() == StateFlags(flags)
        }
        QuickCheck::new()
            .gen(StdGen::new(self::rand::thread_rng(), 10_000))
            .quickcheck(p as fn(Vec<u32>, u8) -> bool);
    }

    #[test]
    fn prop_read_write_u32() {
        fn p(n: u32) -> bool {
            let mut buf = vec![];
            write_varu32(&mut buf, n);
            let (got, nread) = read_varu32(&buf);
            nread == buf.len() && got == n
        }
        quickcheck(p as fn(u32) -> bool);
    }

    #[test]
    fn prop_read_write_i32() {
        fn p(n: i32) -> bool {
            let mut buf = vec![];
            write_vari32(&mut buf, n);
            let (got, nread) = read_vari32(&buf);
            nread == buf.len() && got == n
        }
        quickcheck(p as fn(i32) -> bool);
    }
}
#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), deny(warnings))] // Forbid warnings in release builds
#![warn(clippy::all, rust_2018_idioms)]

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let disable_graphics = option_env!("AF_DISABLE_GRAPHICS");
    if disable_graphics.is_none() || !disable_graphics.unwrap().eq("1"){
        panic!("Please set AF_DISABLE_GRAPHICS=1")
    }
    let app = mosaic::MosaicApp::default();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native("Mosaic",native_options, Box::new(|_| Box::new(app)));
}
use std::fmt::{Display, Formatter};

// Wrapper to avoid unnecessary branching when input doesn't have ANSI escape sequences.
pub struct AnsiStyle {
    attributes: Option<Attributes>,
}

impl AnsiStyle {
    pub fn new() -> Self {
        AnsiStyle { attributes: None }
    }

    pub fn update(&mut self, sequence: &str) -> bool {
        match &mut self.attributes {
            Some(a) => a.update(sequence),
            None => {
                self.attributes = Some(Attributes::new());
                self.attributes.as_mut().unwrap().update(sequence)
            }
        }
    }
}

impl Display for AnsiStyle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.attributes {
            Some(ref a) => a.fmt(f),
            None => Ok(()),
        }
    }
}

struct Attributes {
    foreground: String,
    background: String,
    underlined: String,

    /// The character set to use.
    /// REGEX: `\^[()][AB0-3]`
    charset: String,

    /// A buffer for unknown sequences.
    unknown_buffer: String,

    /// ON:  ^[1m
    /// OFF: ^[22m
    bold: String,

    /// ON:  ^[2m
    /// OFF: ^[22m
    dim: String,

    /// ON:  ^[4m
    /// OFF: ^[24m
    underline: String,

    /// ON:  ^[3m
    /// OFF: ^[23m
    italic: String,

    /// ON:  ^[9m
    /// OFF: ^[29m
    strike: String,
}

impl Attributes {
    pub fn new() -> Self {
        Attributes {
            foreground: "".to_owned(),
            background: "".to_owned(),
            underlined: "".to_owned(),
            charset: "".to_owned(),
            unknown_buffer: "".to_owned(),
            bold: "".to_owned(),
            dim: "".to_owned(),
            underline: "".to_owned(),
            italic: "".to_owned(),
            strike: "".to_owned(),
        }
    }

    /// Update the attributes with an escape sequence.
    /// Returns `false` if the sequence is unsupported.
    pub fn update(&mut self, sequence: &str) -> bool {
        let mut chars = sequence.char_indices().skip(1);

        if let Some((_, t)) = chars.next() {
            match t {
                '(' => self.update_with_charset('(', chars.map(|(_, c)| c)),
                ')' => self.update_with_charset(')', chars.map(|(_, c)| c)),
                '[' => {
                    if let Some((i, last)) = chars.last() {
                        // SAFETY: Always starts with ^[ and ends with m.
                        self.update_with_csi(last, &sequence[2..i])
                    } else {
                        false
                    }
                }
                _ => self.update_with_unsupported(sequence),
            }
        } else {
            false
        }
    }

    fn sgr_reset(&mut self) {
        self.foreground.clear();
        self.background.clear();
        self.underlined.clear();
        self.bold.clear();
        self.dim.clear();
        self.underline.clear();
        self.italic.clear();
        self.strike.clear();
    }

    fn update_with_sgr(&mut self, parameters: &str) -> bool {
        let mut iter = parameters
            .split(';')
            .map(|p| if p.is_empty() { "0" } else { p })
            .map(|p| p.parse::<u16>())
            .map(|p| p.unwrap_or(0)); // Treat errors as 0.

        while let Some(p) = iter.next() {
            match p {
                0 => self.sgr_reset(),
                1 => self.bold = format!("\x1B[{}m", parameters),
                2 => self.dim = format!("\x1B[{}m", parameters),
                3 => self.italic = format!("\x1B[{}m", parameters),
                4 => self.underline = format!("\x1B[{}m", parameters),
                23 => self.italic.clear(),
                24 => self.underline.clear(),
                22 => {
                    self.bold.clear();
                    self.dim.clear();
                }
                30..=39 => self.foreground = Self::parse_color(p, &mut iter),
                40..=49 => self.background = Self::parse_color(p, &mut iter),
                58..=59 => self.underlined = Self::parse_color(p, &mut iter),
                90..=97 => self.foreground = Self::parse_color(p, &mut iter),
                100..=107 => self.foreground = Self::parse_color(p, &mut iter),
                _ => {
                    // Unsupported SGR sequence.
                    // Be compatible and pretend one just wasn't was provided.
                }
            }
        }

        true
    }

    fn update_with_csi(&mut self, finalizer: char, sequence: &str) -> bool {
        if finalizer == 'm' {
            self.update_with_sgr(sequence)
        } else {
            false
        }
    }

    fn update_with_unsupported(&mut self, sequence: &str) -> bool {
        self.unknown_buffer.push_str(sequence);
        false
    }

    fn update_with_charset(&mut self, kind: char, set: impl Iterator<Item = char>) -> bool {
        self.charset = format!("\x1B{}{}", kind, set.take(1).collect::<String>());
        true
    }

    fn parse_color(color: u16, parameters: &mut dyn Iterator<Item = u16>) -> String {
        match color % 10 {
            8 => match parameters.next() {
                Some(5) /* 256-color */ => format!("\x1B[{};5;{}m", color, join(";", 1, parameters)),
                Some(2) /* 24-bit color */ => format!("\x1B[{};2;{}m", color, join(";", 3, parameters)),
                Some(c) => format!("\x1B[{};{}m", color, c),
                _ => "".to_owned(),
            },
            9 => "".to_owned(),
            _ => format!("\x1B[{}m", color),
        }
    }
}

impl Display for Attributes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}{}{}{}{}{}{}{}",
            self.foreground,
            self.background,
            self.underlined,
            self.charset,
            self.bold,
            self.dim,
            self.underline,
            self.italic,
            self.strike,
        )
    }
}

fn join(
    delimiter: &str,
    limit: usize,
    iterator: &mut dyn Iterator<Item = impl ToString>,
) -> String {
    iterator
        .take(limit)
        .map(|i| i.to_string())
        .collect::<Vec<String>>()
        .join(delimiter)
}
use std::io::Write;
use std::vec::Vec;

use ansi_term::Colour::{Fixed, Green, Red, Yellow};
use ansi_term::Style;

use bytesize::ByteSize;

use console::AnsiCodeIterator;

use syntect::easy::HighlightLines;
use syntect::highlighting::Color;
use syntect::highlighting::Theme;
use syntect::parsing::SyntaxSet;

use content_inspector::ContentType;

use encoding::all::{UTF_16BE, UTF_16LE};
use encoding::{DecoderTrap, Encoding};

use unicode_width::UnicodeWidthChar;

use crate::assets::{HighlightingAssets, SyntaxReferenceInSet};
use crate::config::Config;
#[cfg(feature = "git")]
use crate::decorations::LineChangesDecoration;
use crate::decorations::{Decoration, GridBorderDecoration, LineNumberDecoration};
#[cfg(feature = "git")]
use crate::diff::LineChanges;
use crate::error::*;
use crate::input::OpenedInput;
use crate::line_range::RangeCheckResult;
use crate::preprocessor::{expand_tabs, replace_nonprintable};
use crate::style::StyleComponent;
use crate::terminal::{as_terminal_escaped, to_ansi_color};
use crate::vscreen::AnsiStyle;
use crate::wrapping::WrappingMode;

pub(crate) trait Printer {
    fn print_header(
        &mut self,
        handle: &mut dyn Write,
        input: &OpenedInput,
        add_header_padding: bool,
    ) -> Result<()>;
    fn print_footer(&mut self, handle: &mut dyn Write, input: &OpenedInput) -> Result<()>;

    fn print_snip(&mut self, handle: &mut dyn Write) -> Result<()>;

    fn print_line(
        &mut self,
        out_of_range: bool,
        handle: &mut dyn Write,
        line_number: usize,
        line_buffer: &[u8],
    ) -> Result<()>;
}

pub struct SimplePrinter<'a> {
    config: &'a Config<'a>,
}

impl<'a> SimplePrinter<'a> {
    pub fn new(config: &'a Config) -> Self {
        SimplePrinter { config }
    }
}

impl<'a> Printer for SimplePrinter<'a> {
    fn print_header(
        &mut self,
        _handle: &mut dyn Write,
        _input: &OpenedInput,
        _add_header_padding: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn print_footer(&mut self, _handle: &mut dyn Write, _input: &OpenedInput) -> Result<()> {
        Ok(())
    }

    fn print_snip(&mut self, _handle: &mut dyn Write) -> Result<()> {
        Ok(())
    }

    fn print_line(
        &mut self,
        out_of_range: bool,
        handle: &mut dyn Write,
        _line_number: usize,
        line_buffer: &[u8],
    ) -> Result<()> {
        if !out_of_range {
            if self.config.show_nonprintable {
                let line = replace_nonprintable(line_buffer, self.config.tab_width);
                write!(handle, "{}", line)?;
            } else {
                handle.write_all(line_buffer)?
            };
        }
        Ok(())
    }
}

struct HighlighterFromSet<'a> {
    highlighter: HighlightLines<'a>,
    syntax_set: &'a SyntaxSet,
}

impl<'a> HighlighterFromSet<'a> {
    fn new(syntax_in_set: SyntaxReferenceInSet<'a>, theme: &'a Theme) -> Self {
        Self {
            highlighter: HighlightLines::new(syntax_in_set.syntax, theme),
            syntax_set: syntax_in_set.syntax_set,
        }
    }
}

pub(crate) struct InteractivePrinter<'a> {
    colors: Colors,
    config: &'a Config<'a>,
    decorations: Vec<Box<dyn Decoration>>,
    panel_width: usize,
    ansi_style: AnsiStyle,
    content_type: Option<ContentType>,
    #[cfg(feature = "git")]
    pub line_changes: &'a Option<LineChanges>,
    highlighter_from_set: Option<HighlighterFromSet<'a>>,
    background_color_highlight: Option<Color>,
}

impl<'a> InteractivePrinter<'a> {
    pub(crate) fn new(
        config: &'a Config,
        assets: &'a HighlightingAssets,
        input: &mut OpenedInput,
        #[cfg(feature = "git")] line_changes: &'a Option<LineChanges>,
    ) -> Result<Self> {
        let theme = assets.get_theme(&config.theme);

        let background_color_highlight = theme.settings.line_highlight;

        let colors = if config.colored_output {
            Colors::colored(theme, config.true_color)
        } else {
            Colors::plain()
        };

        // Create decorations.
        let mut decorations: Vec<Box<dyn Decoration>> = Vec::new();

        if config.style_components.numbers() {
            decorations.push(Box::new(LineNumberDecoration::new(&colors)));
        }

        #[cfg(feature = "git")]
        {
            if config.style_components.changes() {
                decorations.push(Box::new(LineChangesDecoration::new(&colors)));
            }
        }

        let mut panel_width: usize =
            decorations.len() + decorations.iter().fold(0, |a, x| a + x.width());

        // The grid border decoration isn't added until after the panel_width calculation, since the
        // print_horizontal_line, print_header, and print_footer functions all assume the panel
        // width is without the grid border.
        if config.style_components.grid() && !decorations.is_empty() {
            decorations.push(Box::new(GridBorderDecoration::new(&colors)));
        }

        // Disable the panel if the terminal is too small (i.e. can't fit 5 characters with the
        // panel showing).
        if config.term_width
            < (decorations.len() + decorations.iter().fold(0, |a, x| a + x.width())) + 5
        {
            decorations.clear();
            panel_width = 0;
        }

        let highlighter_from_set = if input
            .reader
            .content_type
            .map_or(false, |c| c.is_binary() && !config.show_nonprintable)
        {
            None
        } else {
            // Determine the type of syntax for highlighting
            let syntax_in_set =
                match assets.get_syntax(config.language, input, &config.syntax_mapping) {
                    Ok(syntax_in_set) => syntax_in_set,
                    Err(Error::UndetectedSyntax(_)) => assets
                        .find_syntax_by_name("Plain Text")?
                        .expect("A plain text syntax is available"),
                    Err(e) => return Err(e),
                };

            Some(HighlighterFromSet::new(syntax_in_set, theme))
        };

        Ok(InteractivePrinter {
            panel_width,
            colors,
            config,
            decorations,
            content_type: input.reader.content_type,
            ansi_style: AnsiStyle::new(),
            #[cfg(feature = "git")]
            line_changes,
            highlighter_from_set,
            background_color_highlight,
        })
    }

    fn print_horizontal_line_term(&mut self, handle: &mut dyn Write, style: Style) -> Result<()> {
        writeln!(
            handle,
            "{}",
            style.paint("─".repeat(self.config.term_width))
        )?;
        Ok(())
    }

    fn print_horizontal_line(&mut self, handle: &mut dyn Write, grid_char: char) -> Result<()> {
        if self.panel_width == 0 {
            self.print_horizontal_line_term(handle, self.colors.grid)?;
        } else {
            let hline = "─".repeat(self.config.term_width - (self.panel_width + 1));
            let hline = format!("{}{}{}", "─".repeat(self.panel_width), grid_char, hline);
            writeln!(handle, "{}", self.colors.grid.paint(hline))?;
        }

        Ok(())
    }

    fn create_fake_panel(&self, text: &str) -> String {
        if self.panel_width == 0 {
            return "".to_string();
        }

        let text_truncated: String = text.chars().take(self.panel_width - 1).collect();
        let text_filled: String = format!(
            "{}{}",
            text_truncated,
            " ".repeat(self.panel_width - 1 - text_truncated.len())
        );
        if self.config.style_components.grid() {
            format!("{} │ ", text_filled)
        } else {
            text_filled
        }
    }

    fn print_header_component_indent(&mut self, handle: &mut dyn Write) -> std::io::Result<()> {
        if self.config.style_components.grid() {
            write!(
                handle,
                "{}{}",
                " ".repeat(self.panel_width),
                self.colors
                    .grid
                    .paint(if self.panel_width > 0 { "│ " } else { "" }),
            )
        } else {
            write!(handle, "{}", " ".repeat(self.panel_width))
        }
    }

    fn preprocess(&self, text: &str, cursor: &mut usize) -> String {
        if self.config.tab_width > 0 {
            return expand_tabs(text, self.config.tab_width, cursor);
        }

        *cursor += text.len();
        text.to_string()
    }
}

impl<'a> Printer for InteractivePrinter<'a> {
    fn print_header(
        &mut self,
        handle: &mut dyn Write,
        input: &OpenedInput,
        add_header_padding: bool,
    ) -> Result<()> {
        if add_header_padding && self.config.style_components.rule() {
            self.print_horizontal_line_term(handle, self.colors.rule)?;
        }

        if !self.config.style_components.header() {
            if Some(ContentType::BINARY) == self.content_type && !self.config.show_nonprintable {
                writeln!(
                    handle,
                    "{}: Binary content from {} will not be printed to the terminal \
                     (but will be present if the output of 'bat' is piped). You can use 'bat -A' \
                     to show the binary file contents.",
                    Yellow.paint("[bat warning]"),
                    input.description.summary(),
                )?;
            } else if self.config.style_components.grid() {
                self.print_horizontal_line(handle, '┬')?;
            }
            return Ok(());
        }

        let mode = match self.content_type {
            Some(ContentType::BINARY) => "   <BINARY>",
            Some(ContentType::UTF_16LE) => "   <UTF-16LE>",
            Some(ContentType::UTF_16BE) => "   <UTF-16BE>",
            None => "   <EMPTY>",
            _ => "",
        };

        let description = &input.description;
        let metadata = &input.metadata;

        // We use this iterator to have a deterministic order for
        // header components. HashSet has arbitrary order, but Vec is ordered.
        let header_components: Vec<StyleComponent> = [
            (
                StyleComponent::HeaderFilename,
                self.config.style_components.header_filename(),
            ),
            (
                StyleComponent::HeaderFilesize,
                self.config.style_components.header_filesize(),
            ),
        ]
        .iter()
        .filter(|(_, is_enabled)| *is_enabled)
        .map(|(component, _)| *component)
        .collect();

        // Print the cornering grid before the first header component
        if self.config.style_components.grid() {
            self.print_horizontal_line(handle, '┬')?;
        } else {
            // Only pad space between files, if we haven't already drawn a horizontal rule
            if add_header_padding && !self.config.style_components.rule() {
                writeln!(handle)?;
            }
        }

        header_components.iter().try_for_each(|component| {
            self.print_header_component_indent(handle)?;

            match component {
                StyleComponent::HeaderFilename => writeln!(
                    handle,
                    "{}{}{}",
                    description
                        .kind()
                        .map(|kind| format!("{}: ", kind))
                        .unwrap_or_else(|| "".into()),
                    self.colors.header_value.paint(description.title()),
                    mode
                ),

                StyleComponent::HeaderFilesize => {
                    let bsize = metadata
                        .size
                        .map(|s| format!("{}", ByteSize(s)))
                        .unwrap_or_else(|| "-".into());
                    writeln!(handle, "Size: {}", self.colors.header_value.paint(bsize))
                }
                _ => Ok(()),
            }
        })?;

        if self.config.style_components.grid() {
            if self.content_type.map_or(false, |c| c.is_text()) || self.config.show_nonprintable {
                self.print_horizontal_line(handle, '┼')?;
            } else {
                self.print_horizontal_line(handle, '┴')?;
            }
        }

        Ok(())
    }

    fn print_footer(&mut self, handle: &mut dyn Write, _input: &OpenedInput) -> Result<()> {
        if self.config.style_components.grid()
            && (self.content_type.map_or(false, |c| c.is_text()) || self.config.show_nonprintable)
        {
            self.print_horizontal_line(handle, '┴')
        } else {
            Ok(())
        }
    }

    fn print_snip(&mut self, handle: &mut dyn Write) -> Result<()> {
        let panel = self.create_fake_panel(" ...");
        let panel_count = panel.chars().count();

        let title = "8<";
        let title_count = title.chars().count();

        let snip_left = "─ ".repeat((self.config.term_width - panel_count - (title_count / 2)) / 4);
        let snip_left_count = snip_left.chars().count(); // Can't use .len() with Unicode.

        let snip_right =
            " ─".repeat((self.config.term_width - panel_count - snip_left_count - title_count) / 2);

        writeln!(
            handle,
            "{}",
            self.colors
                .grid
                .paint(format!("{}{}{}{}", panel, snip_left, title, snip_right))
        )?;

        Ok(())
    }

    fn print_line(
        &mut self,
        out_of_range: bool,
        handle: &mut dyn Write,
        line_number: usize,
        line_buffer: &[u8],
    ) -> Result<()> {
        let line = if self.config.show_nonprintable {
            replace_nonprintable(line_buffer, self.config.tab_width)
        } else {
            match self.content_type {
                Some(ContentType::BINARY) | None => {
                    return Ok(());
                }
                Some(ContentType::UTF_16LE) => UTF_16LE
                    .decode(line_buffer, DecoderTrap::Replace)
                    .map_err(|_| "Invalid UTF-16LE")?,
                Some(ContentType::UTF_16BE) => UTF_16BE
                    .decode(line_buffer, DecoderTrap::Replace)
                    .map_err(|_| "Invalid UTF-16BE")?,
                _ => String::from_utf8_lossy(line_buffer).to_string(),
            }
        };

        let regions = {
            let highlighter_from_set = match self.highlighter_from_set {
                Some(ref mut highlighter_from_set) => highlighter_from_set,
                _ => {
                    return Ok(());
                }
            };
            highlighter_from_set
                .highlighter
                .highlight(&line, highlighter_from_set.syntax_set)
        };

        if out_of_range {
            return Ok(());
        }

        let mut cursor: usize = 0;
        let mut cursor_max: usize = self.config.term_width;
        let mut cursor_total: usize = 0;
        let mut panel_wrap: Option<String> = None;

        // Line highlighting
        let highlight_this_line =
            self.config.highlighted_lines.0.check(line_number) == RangeCheckResult::InRange;

        if highlight_this_line && self.config.theme == "ansi" {
            self.ansi_style.update("^[4m");
        }

        let background_color = self
            .background_color_highlight
            .filter(|_| highlight_this_line);

        // Line decorations.
        if self.panel_width > 0 {
            let decorations = self
                .decorations
                .iter()
                .map(|d| d.generate(line_number, false, self));

            for deco in decorations {
                write!(handle, "{} ", deco.text)?;
                cursor_max -= deco.width + 1;
            }
        }

        // Line contents.
        if matches!(self.config.wrapping_mode, WrappingMode::NoWrapping(_)) {
            let true_color = self.config.true_color;
            let colored_output = self.config.colored_output;
            let italics = self.config.use_italic_text;

            for &(style, region) in &regions {
                let ansi_iterator = AnsiCodeIterator::new(region);
                for chunk in ansi_iterator {
                    match chunk {
                        // ANSI escape passthrough.
                        (ansi, true) => {
                            self.ansi_style.update(ansi);
                            write!(handle, "{}", ansi)?;
                        }

                        // Regular text.
                        (text, false) => {
                            let text = &*self.preprocess(text, &mut cursor_total);
                            let text_trimmed = text.trim_end_matches(|c| c == '\r' || c == '\n');

                            write!(
                                handle,
                                "{}",
                                as_terminal_escaped(
                                    style,
                                    &format!("{}{}", self.ansi_style, text_trimmed),
                                    true_color,
                                    colored_output,
                                    italics,
                                    background_color
                                )
                            )?;

                            if text.len() != text_trimmed.len() {
                                if let Some(background_color) = background_color {
                                    let ansi_style = Style {
                                        background: to_ansi_color(background_color, true_color),
                                        ..Default::default()
                                    };

                                    let width = if cursor_total <= cursor_max {
                                        cursor_max - cursor_total + 1
                                    } else {
                                        0
                                    };
                                    write!(handle, "{}", ansi_style.paint(" ".repeat(width)))?;
                                }
                                write!(handle, "{}", &text[text_trimmed.len()..])?;
                            }
                        }
                    }
                }
            }

            if !self.config.style_components.plain() && line.bytes().next_back() != Some(b'\n') {
                writeln!(handle)?;
            }
        } else {
            for &(style, region) in &regions {
                let ansi_iterator = AnsiCodeIterator::new(region);
                for chunk in ansi_iterator {
                    match chunk {
                        // ANSI escape passthrough.
                        (ansi, true) => {
                            self.ansi_style.update(ansi);
                            write!(handle, "{}", ansi)?;
                        }

                        // Regular text.
                        (text, false) => {
                            let text = self.preprocess(
                                text.trim_end_matches(|c| c == '\r' || c == '\n'),
                                &mut cursor_total,
                            );

                            let mut max_width = cursor_max - cursor;

                            // line buffer (avoid calling write! for every character)
                            let mut line_buf = String::with_capacity(max_width * 4);

                            // Displayed width of line_buf
                            let mut current_width = 0;

                            for c in text.chars() {
                                // calculate the displayed width for next character
                                let cw = c.width().unwrap_or(0);
                                current_width += cw;

                                // if next character cannot be printed on this line,
                                // flush the buffer.
                                if current_width > max_width {
                                    // Generate wrap padding if not already generated.
                                    if panel_wrap.is_none() {
                                        panel_wrap = if self.panel_width > 0 {
                                            Some(format!(
                                                "{} ",
                                                self.decorations
                                                    .iter()
                                                    .map(|d| d
                                                        .generate(line_number, true, self)
                                                        .text)
                                                    .collect::<Vec<String>>()
                                                    .join(" ")
                                            ))
                                        } else {
                                            Some("".to_string())
                                        }
                                    }

                                    // It wraps.
                                    write!(
                                        handle,
                                        "{}\n{}",
                                        as_terminal_escaped(
                                            style,
                                            &*format!("{}{}", self.ansi_style, line_buf),
                                            self.config.true_color,
                                            self.config.colored_output,
                                            self.config.use_italic_text,
                                            background_color
                                        ),
                                        panel_wrap.clone().unwrap()
                                    )?;

                                    cursor = 0;
                                    max_width = cursor_max;

                                    line_buf.clear();
                                    current_width = cw;
                                }

                                line_buf.push(c);
                            }

                            // flush the buffer
                            cursor += current_width;
                            write!(
                                handle,
                                "{}",
                                as_terminal_escaped(
                                    style,
                                    &*format!("{}{}", self.ansi_style, line_buf),
                                    self.config.true_color,
                                    self.config.colored_output,
                                    self.config.use_italic_text,
                                    background_color
                                )
                            )?;
                        }
                    }
                }
            }

            if let Some(background_color) = background_color {
                let ansi_style = Style {
                    background: to_ansi_color(background_color, self.config.true_color),
                    ..Default::default()
                };

                write!(
                    handle,
                    "{}",
                    ansi_style.paint(" ".repeat(cursor_max - cursor))
                )?;
            }
            writeln!(handle)?;
        }

        if highlight_this_line && self.config.theme == "ansi" {
            self.ansi_style.update("^[24m");
            write!(handle, "\x1B[24m")?;
        }

        Ok(())
    }
}

const DEFAULT_GUTTER_COLOR: u8 = 238;

#[derive(Debug, Default)]
pub struct Colors {
    pub grid: Style,
    pub rule: Style,
    pub header_value: Style,
    pub git_added: Style,
    pub git_removed: Style,
    pub git_modified: Style,
    pub line_number: Style,
}

impl Colors {
    fn plain() -> Self {
        Colors::default()
    }

    fn colored(theme: &Theme, true_color: bool) -> Self {
        let gutter_style = Style {
            foreground: match theme.settings.gutter_foreground {
                // If the theme provides a gutter foreground color, use it.
                // Note: It might be the special value #00000001, in which case
                // to_ansi_color returns None and we use an empty Style
                // (resulting in the terminal's default foreground color).
                Some(c) => to_ansi_color(c, true_color),
                // Otherwise, use a specific fallback color.
                None => Some(Fixed(DEFAULT_GUTTER_COLOR)),
            },
            ..Style::default()
        };

        Colors {
            grid: gutter_style,
            rule: gutter_style,
            header_value: Style::new().bold(),
            git_added: Green.normal(),
            git_removed: Red.normal(),
            git_modified: Yellow.normal(),
            line_number: gutter_style,
        }
    }
}
use crate::connector::Connector;

fn create_connector()-> Connector{
    let user = "rust".to_string();
    let host = "localhost".to_string();
    let database = "rust_test".to_string();
    Connector::new(user, host, database)
}

#[test]
pub fn test_connector() {
    let connector = create_connector();
    let mut client = connector.client().unwrap();
    let err = client.batch_execute("
        CREATE TABLE IF NOT EXISTS test (
            id              SERIAL PRIMARY KEY,
            data            VARCHAR NOT NULL
            )
    ").unwrap();
}

#[cfg(test)]
mod pg_connection{
    use crate::connection::PgConnection;
    use crate::connection::Connection;
    use crate::models::Model;

    #[test]
    pub fn test_pg_connection(){
        let client = super::create_connector().client().unwrap();
        let pg_connection = PgConnection::new(client);
    }

    struct TestModel{
        id:u32,
    }
    impl Model for TestModel{
    }
    #[test]
    pub fn test_pg_create_model(){
        let client = super::create_connector().client().unwrap();
        let pg_connection = PgConnection::new(client);
        pg_connection.register_model(TestModel {id:10});
    }
}
use s3::creds::Credentials;
use s3::region::Region;

pub mod files;
pub mod inode;
pub mod inodetree;
pub mod datasource;

pub struct Storage {
    pub name: String,
    pub region: Region,
    pub credentials: Credentials,
    pub bucket: String,
}

impl Storage {
    pub fn new(name:String, region:Region,bucket:String) -> Self{
        Self{
            name,
            region,
            credentials: Credentials::default_blocking().unwrap(),
            bucket,
        }
    }
}
// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ptr;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;
use std::thread::{self, ThreadId};

use crate::error::{Error, Result};
use crate::grpc_sys::{self, gpr_clock_type, grpc_completion_queue};
use crate::task::UnfinishedWork;

pub use crate::grpc_sys::grpc_completion_type as EventType;
pub use crate::grpc_sys::grpc_event as Event;

/// `CompletionQueueHandle` enable notification of the completion of asynchronous actions.
pub struct CompletionQueueHandle {
    cq: *mut grpc_completion_queue,
    // When `ref_cnt` < 0, a shutdown is pending, completion queue should not
    // accept requests anymore; when `ref_cnt` == 0, completion queue should
    // be shutdown; When `ref_cnt` > 0, completion queue can accept requests
    // and should not be shutdown.
    ref_cnt: AtomicIsize,
}

unsafe impl Sync for CompletionQueueHandle {}
unsafe impl Send for CompletionQueueHandle {}

impl CompletionQueueHandle {
    pub fn new() -> CompletionQueueHandle {
        CompletionQueueHandle {
            cq: unsafe { grpc_sys::grpc_completion_queue_create_for_next(ptr::null_mut()) },
            ref_cnt: AtomicIsize::new(1),
        }
    }

    fn add_ref(&self) -> Result<()> {
        loop {
            let cnt = self.ref_cnt.load(Ordering::SeqCst);
            if cnt <= 0 {
                // `shutdown` has been called, reject any requests.
                return Err(Error::QueueShutdown);
            }
            let new_cnt = cnt + 1;
            if cnt
                == self
                    .ref_cnt
                    .compare_and_swap(cnt, new_cnt, Ordering::SeqCst)
            {
                return Ok(());
            }
        }
    }

    fn unref(&self) {
        let shutdown = loop {
            let cnt = self.ref_cnt.load(Ordering::SeqCst);
            // If `shutdown` is not called, `cnt` > 0, so minus 1 to unref.
            // If `shutdown` is called, `cnt` < 0, so plus 1 to unref.
            let new_cnt = cnt - cnt.signum();
            if cnt
                == self
                    .ref_cnt
                    .compare_and_swap(cnt, new_cnt, Ordering::SeqCst)
            {
                break new_cnt == 0;
            }
        };
        if shutdown {
            unsafe {
                grpc_sys::grpc_completion_queue_shutdown(self.cq);
            }
        }
    }

    fn shutdown(&self) {
        let shutdown = loop {
            let cnt = self.ref_cnt.load(Ordering::SeqCst);
            if cnt <= 0 {
                // `shutdown` is called, skipped.
                return;
            }
            // Make cnt negative to indicate that `shutdown` has been called.
            // Because `cnt` is initialized to 1, so minus 1 to make it reach
            // toward 0. That is `new_cnt = -(cnt - 1) = -cnt + 1`.
            let new_cnt = -cnt + 1;
            if cnt
                == self
                    .ref_cnt
                    .compare_and_swap(cnt, new_cnt, Ordering::SeqCst)
            {
                break new_cnt == 0;
            }
        };
        if shutdown {
            unsafe {
                grpc_sys::grpc_completion_queue_shutdown(self.cq);
            }
        }
    }
}

impl Drop for CompletionQueueHandle {
    fn drop(&mut self) {
        unsafe { grpc_sys::grpc_completion_queue_destroy(self.cq) }
    }
}

pub struct CompletionQueueRef<'a> {
    queue: &'a CompletionQueue,
}

impl<'a> CompletionQueueRef<'a> {
    pub fn as_ptr(&self) -> *mut grpc_completion_queue {
        self.queue.handle.cq
    }
}

impl<'a> Drop for CompletionQueueRef<'a> {
    fn drop(&mut self) {
        self.queue.handle.unref();
    }
}

/// `WorkQueue` stores the unfinished work of a completion queue.
///
/// Every completion queue has a work queue, and every work queue belongs
/// to exact one completion queue. `WorkQueue` is a short path for future
/// notifications. When a future is ready to be polled, there are two way
/// to notify it.
/// 1. If it's in the same thread where the future is spawned, the future
///    will be pushed into `WorkQueue` and be polled when current call tag
///    is handled;
/// 2. If not, the future will be wrapped as a call tag and pushed into
///    completion queue and finally popped at the call to `grpc_completion_queue_next`.
pub struct WorkQueue {
    id: ThreadId,
    pending_work: UnsafeCell<VecDeque<UnfinishedWork>>,
}

unsafe impl Sync for WorkQueue {}
unsafe impl Send for WorkQueue {}

const QUEUE_CAPACITY: usize = 4096;

impl WorkQueue {
    pub fn new() -> WorkQueue {
        WorkQueue {
            id: std::thread::current().id(),
            pending_work: UnsafeCell::new(VecDeque::with_capacity(QUEUE_CAPACITY)),
        }
    }

    /// Pushes an unfinished work into the inner queue.
    ///
    /// If the method is not called from the same thread where it's created,
    /// the work will returned and no work is pushed.
    pub fn push_work(&self, work: UnfinishedWork) -> Option<UnfinishedWork> {
        if self.id == thread::current().id() {
            unsafe { &mut *self.pending_work.get() }.push_back(work);
            None
        } else {
            Some(work)
        }
    }

    /// Pops one unfinished work.
    ///
    /// It should only be called from the same thread where the queue is created.
    /// Otherwise it leads to undefined behavior.
    pub unsafe fn pop_work(&self) -> Option<UnfinishedWork> {
        let queue = &mut *self.pending_work.get();
        if queue.capacity() > QUEUE_CAPACITY && queue.len() < queue.capacity() / 2 {
            queue.shrink_to_fit();
        }
        { &mut *self.pending_work.get() }.pop_back()
    }
}

#[derive(Clone)]
pub struct CompletionQueue {
    handle: Arc<CompletionQueueHandle>,
    pub(crate) worker: Arc<WorkQueue>,
}

impl CompletionQueue {
    pub fn new(handle: Arc<CompletionQueueHandle>, worker: Arc<WorkQueue>) -> CompletionQueue {
        CompletionQueue { handle, worker }
    }

    /// Blocks until an event is available, the completion queue is being shut down.
    pub fn next(&self) -> Event {
        unsafe {
            let inf = grpc_sys::gpr_inf_future(gpr_clock_type::GPR_CLOCK_REALTIME);
            grpc_sys::grpc_completion_queue_next(self.handle.cq, inf, ptr::null_mut())
        }
    }

    pub fn borrow(&self) -> Result<CompletionQueueRef<'_>> {
        self.handle.add_ref()?;
        Ok(CompletionQueueRef { queue: self })
    }

    /// Begin destruction of a completion queue.
    ///
    /// Once all possible events are drained then `next()` will start to produce
    /// `Event::QueueShutdown` events only.
    pub fn shutdown(&self) {
        self.handle.shutdown()
    }

    pub fn worker_id(&self) -> ThreadId {
        self.worker.id
    }
}
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "perf-literal")]
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use syntax::hir::literal::Literals;
use syntax::hir::Hir;
use syntax::ParserBuilder;

use backtrack;
use cache::{Cached, CachedGuard};
use compile::Compiler;
#[cfg(feature = "perf-dfa")]
use dfa;
use error::Error;
use input::{ByteInput, CharInput};
use literal::LiteralSearcher;
use pikevm;
use prog::Program;
use re_builder::RegexOptions;
use re_bytes;
use re_set;
use re_trait::{Locations, RegularExpression, Slot};
use re_unicode;
use utf8::next_utf8;

/// `Exec` manages the execution of a regular expression.
///
/// In particular, this manages the various compiled forms of a single regular
/// expression and the choice of which matching engine to use to execute a
/// regular expression.
pub struct Exec {
    /// All read only state.
    ro: Arc<ExecReadOnly>,
    /// Caches for the various matching engines.
    cache: Cached<ProgramCache>,
}

/// `ExecNoSync` is like `Exec`, except it embeds a reference to a cache. This
/// means it is no longer Sync, but we can now avoid the overhead of
/// synchronization to fetch the cache.
#[derive(Debug)]
pub struct ExecNoSync<'c> {
    /// All read only state.
    ro: &'c Arc<ExecReadOnly>,
    /// Caches for the various matching engines.
    cache: CachedGuard<'c, ProgramCache>,
}

/// `ExecNoSyncStr` is like `ExecNoSync`, but matches on &str instead of &[u8].
pub struct ExecNoSyncStr<'c>(ExecNoSync<'c>);

/// `ExecReadOnly` comprises all read only state for a regex. Namely, all such
/// state is determined at compile time and never changes during search.
#[derive(Debug)]
struct ExecReadOnly {
    /// The original regular expressions given by the caller to compile.
    res: Vec<String>,
    /// A compiled program that is used in the NFA simulation and backtracking.
    /// It can be byte-based or Unicode codepoint based.
    ///
    /// N.B. It is not possibly to make this byte-based from the public API.
    /// It is only used for testing byte based programs in the NFA simulations.
    nfa: Program,
    /// A compiled byte based program for DFA execution. This is only used
    /// if a DFA can be executed. (Currently, only word boundary assertions are
    /// not supported.) Note that this program contains an embedded `.*?`
    /// preceding the first capture group, unless the regex is anchored at the
    /// beginning.
    dfa: Program,
    /// The same as above, except the program is reversed (and there is no
    /// preceding `.*?`). This is used by the DFA to find the starting location
    /// of matches.
    dfa_reverse: Program,
    /// A set of suffix literals extracted from the regex.
    ///
    /// Prefix literals are stored on the `Program`, since they are used inside
    /// the matching engines.
    suffixes: LiteralSearcher,
    /// An Aho-Corasick automaton with leftmost-first match semantics.
    ///
    /// This is only set when the entire regex is a simple unanchored
    /// alternation of literals. We could probably use it more circumstances,
    /// but this is already hacky enough in this architecture.
    ///
    /// N.B. We use u32 as a state ID representation under the assumption that
    /// if we were to exhaust the ID space, we probably would have long
    /// surpassed the compilation size limit.
    #[cfg(feature = "perf-literal")]
    ac: Option<AhoCorasick<u32>>,
    /// match_type encodes as much upfront knowledge about how we're going to
    /// execute a search as possible.
    match_type: MatchType,
}

/// Facilitates the construction of an executor by exposing various knobs
/// to control how a regex is executed and what kinds of resources it's
/// permitted to use.
pub struct ExecBuilder {
    options: RegexOptions,
    match_type: Option<MatchType>,
    bytes: bool,
    only_utf8: bool,
}

/// Parsed represents a set of parsed regular expressions and their detected
/// literals.
struct Parsed {
    exprs: Vec<Hir>,
    prefixes: Literals,
    suffixes: Literals,
    bytes: bool,
}

impl ExecBuilder {
    /// Create a regex execution builder.
    ///
    /// This uses default settings for everything except the regex itself,
    /// which must be provided. Further knobs can be set by calling methods,
    /// and then finally, `build` to actually create the executor.
    pub fn new(re: &str) -> Self {
        Self::new_many(&[re])
    }

    /// Like new, but compiles the union of the given regular expressions.
    ///
    /// Note that when compiling 2 or more regular expressions, capture groups
    /// are completely unsupported. (This means both `find` and `captures`
    /// wont work.)
    pub fn new_many<I, S>(res: I) -> Self
    where
        S: AsRef<str>,
        I: IntoIterator<Item = S>,
    {
        let mut opts = RegexOptions::default();
        opts.pats = res.into_iter().map(|s| s.as_ref().to_owned()).collect();
        Self::new_options(opts)
    }

    /// Create a regex execution builder.
    pub fn new_options(opts: RegexOptions) -> Self {
        ExecBuilder {
            options: opts,
            match_type: None,
            bytes: false,
            only_utf8: true,
        }
    }

    /// Set the matching engine to be automatically determined.
    ///
    /// This is the default state and will apply whatever optimizations are
    /// possible, such as running a DFA.
    ///
    /// This overrides whatever was previously set via the `nfa` or
    /// `bounded_backtracking` methods.
    pub fn automatic(mut self) -> Self {
        self.match_type = None;
        self
    }

    /// Sets the matching engine to use the NFA algorithm no matter what
    /// optimizations are possible.
    ///
    /// This overrides whatever was previously set via the `automatic` or
    /// `bounded_backtracking` methods.
    pub fn nfa(mut self) -> Self {
        self.match_type = Some(MatchType::Nfa(MatchNfaType::PikeVM));
        self
    }

    /// Sets the matching engine to use a bounded backtracking engine no
    /// matter what optimizations are possible.
    ///
    /// One must use this with care, since the bounded backtracking engine
    /// uses memory proportion to `len(regex) * len(text)`.
    ///
    /// This overrides whatever was previously set via the `automatic` or
    /// `nfa` methods.
    pub fn bounded_backtracking(mut self) -> Self {
        self.match_type = Some(MatchType::Nfa(MatchNfaType::Backtrack));
        self
    }

    /// Compiles byte based programs for use with the NFA matching engines.
    ///
    /// By default, the NFA engines match on Unicode scalar values. They can
    /// be made to use byte based programs instead. In general, the byte based
    /// programs are slower because of a less efficient encoding of character
    /// classes.
    ///
    /// Note that this does not impact DFA matching engines, which always
    /// execute on bytes.
    pub fn bytes(mut self, yes: bool) -> Self {
        self.bytes = yes;
        self
    }

    /// When disabled, the program compiled may match arbitrary bytes.
    ///
    /// When enabled (the default), all compiled programs exclusively match
    /// valid UTF-8 bytes.
    pub fn only_utf8(mut self, yes: bool) -> Self {
        self.only_utf8 = yes;
        self
    }

    /// Set the Unicode flag.
    pub fn unicode(mut self, yes: bool) -> Self {
        self.options.unicode = yes;
        self
    }

    /// Parse the current set of patterns into their AST and extract literals.
    fn parse(&self) -> Result<Parsed, Error> {
        let mut exprs = Vec::with_capacity(self.options.pats.len());
        let mut prefixes = Some(Literals::empty());
        let mut suffixes = Some(Literals::empty());
        let mut bytes = false;
        let is_set = self.options.pats.len() > 1;
        // If we're compiling a regex set and that set has any anchored
        // expressions, then disable all literal optimizations.
        for pat in &self.options.pats {
            let mut parser = ParserBuilder::new()
                .octal(self.options.octal)
                .case_insensitive(self.options.case_insensitive)
                .multi_line(self.options.multi_line)
                .dot_matches_new_line(self.options.dot_matches_new_line)
                .swap_greed(self.options.swap_greed)
                .ignore_whitespace(self.options.ignore_whitespace)
                .unicode(self.options.unicode)
                .allow_invalid_utf8(!self.only_utf8)
                .nest_limit(self.options.nest_limit)
                .build();
            let expr =
                parser.parse(pat).map_err(|e| Error::Syntax(e.to_string()))?;
            bytes = bytes || !expr.is_always_utf8();

            if cfg!(feature = "perf-literal") {
                if !expr.is_anchored_start() && expr.is_any_anchored_start() {
                    // Partial anchors unfortunately make it hard to use
                    // prefixes, so disable them.
                    prefixes = None;
                } else if is_set && expr.is_anchored_start() {
                    // Regex sets with anchors do not go well with literal
                    // optimizations.
                    prefixes = None;
                }
                prefixes = prefixes.and_then(|mut prefixes| {
                    if !prefixes.union_prefixes(&expr) {
                        None
                    } else {
                        Some(prefixes)
                    }
                });

                if !expr.is_anchored_end() && expr.is_any_anchored_end() {
                    // Partial anchors unfortunately make it hard to use
                    // suffixes, so disable them.
                    suffixes = None;
                } else if is_set && expr.is_anchored_end() {
                    // Regex sets with anchors do not go well with literal
                    // optimizations.
                    suffixes = None;
                }
                suffixes = suffixes.and_then(|mut suffixes| {
                    if !suffixes.union_suffixes(&expr) {
                        None
                    } else {
                        Some(suffixes)
                    }
                });
            }
            exprs.push(expr);
        }
        Ok(Parsed {
            exprs: exprs,
            prefixes: prefixes.unwrap_or_else(Literals::empty),
            suffixes: suffixes.unwrap_or_else(Literals::empty),
            bytes: bytes,
        })
    }

    /// Build an executor that can run a regular expression.
    pub fn build(self) -> Result<Exec, Error> {
        // Special case when we have no patterns to compile.
        // This can happen when compiling a regex set.
        if self.options.pats.is_empty() {
            let ro = Arc::new(ExecReadOnly {
                res: vec![],
                nfa: Program::new(),
                dfa: Program::new(),
                dfa_reverse: Program::new(),
                suffixes: LiteralSearcher::empty(),
                #[cfg(feature = "perf-literal")]
                ac: None,
                match_type: MatchType::Nothing,
            });
            return Ok(Exec { ro: ro, cache: Cached::new() });
        }
        let parsed = self.parse()?;
        let mut nfa = Compiler::new()
            .size_limit(self.options.size_limit)
            .bytes(self.bytes || parsed.bytes)
            .only_utf8(self.only_utf8)
            .compile(&parsed.exprs)?;
        let mut dfa = Compiler::new()
            .size_limit(self.options.size_limit)
            .dfa(true)
            .only_utf8(self.only_utf8)
            .compile(&parsed.exprs)?;
        let mut dfa_reverse = Compiler::new()
            .size_limit(self.options.size_limit)
            .dfa(true)
            .only_utf8(self.only_utf8)
            .reverse(true)
            .compile(&parsed.exprs)?;

        #[cfg(feature = "perf-literal")]
        let ac = self.build_aho_corasick(&parsed);
        nfa.prefixes = LiteralSearcher::prefixes(parsed.prefixes);
        dfa.prefixes = nfa.prefixes.clone();
        dfa.dfa_size_limit = self.options.dfa_size_limit;
        dfa_reverse.dfa_size_limit = self.options.dfa_size_limit;

        let mut ro = ExecReadOnly {
            res: self.options.pats,
            nfa: nfa,
            dfa: dfa,
            dfa_reverse: dfa_reverse,
            suffixes: LiteralSearcher::suffixes(parsed.suffixes),
            #[cfg(feature = "perf-literal")]
            ac: ac,
            match_type: MatchType::Nothing,
        };
        ro.match_type = ro.choose_match_type(self.match_type);

        let ro = Arc::new(ro);
        Ok(Exec { ro: ro, cache: Cached::new() })
    }

    #[cfg(feature = "perf-literal")]
    fn build_aho_corasick(&self, parsed: &Parsed) -> Option<AhoCorasick<u32>> {
        if parsed.exprs.len() != 1 {
            return None;
        }
        let lits = match alternation_literals(&parsed.exprs[0]) {
            None => return None,
            Some(lits) => lits,
        };
        // If we have a small number of literals, then let Teddy handle
        // things (see literal/mod.rs).
        if lits.len() <= 32 {
            return None;
        }
        Some(
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostFirst)
                .auto_configure(&lits)
                // We always want this to reduce size, regardless
                // of what auto-configure does.
                .byte_classes(true)
                .build_with_size::<u32, _, _>(&lits)
                // This should never happen because we'd long exceed the
                // compilation limit for regexes first.
                .expect("AC automaton too big"),
        )
    }
}

impl<'c> RegularExpression for ExecNoSyncStr<'c> {
    type Text = str;

    fn slots_len(&self) -> usize {
        self.0.slots_len()
    }

    fn next_after_empty(&self, text: &str, i: usize) -> usize {
        next_utf8(text.as_bytes(), i)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_match_at(&self, text: &str, start: usize) -> Option<usize> {
        self.0.shortest_match_at(text.as_bytes(), start)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_match_at(&self, text: &str, start: usize) -> bool {
        self.0.is_match_at(text.as_bytes(), start)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_at(&self, text: &str, start: usize) -> Option<(usize, usize)> {
        self.0.find_at(text.as_bytes(), start)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn captures_read_at(
        &self,
        locs: &mut Locations,
        text: &str,
        start: usize,
    ) -> Option<(usize, usize)> {
        self.0.captures_read_at(locs, text.as_bytes(), start)
    }
}

impl<'c> RegularExpression for ExecNoSync<'c> {
    type Text = [u8];

    /// Returns the number of capture slots in the regular expression. (There
    /// are two slots for every capture group, corresponding to possibly empty
    /// start and end locations of the capture.)
    fn slots_len(&self) -> usize {
        self.ro.nfa.captures.len() * 2
    }

    fn next_after_empty(&self, _text: &[u8], i: usize) -> usize {
        i + 1
    }

    /// Returns the end of a match location, possibly occurring before the
    /// end location of the correct leftmost-first match.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_match_at(&self, text: &[u8], start: usize) -> Option<usize> {
        if !self.is_anchor_end_match(text) {
            return None;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => {
                self.find_literals(ty, text, start).map(|(_, e)| e)
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa | MatchType::DfaMany => {
                match self.shortest_dfa(text, start) {
                    dfa::Result::Match(end) => Some(end),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.shortest_nfa(text, start),
                }
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match dfa::Fsm::reverse(
                    &self.ro.dfa_reverse,
                    self.cache.value(),
                    true,
                    &text[start..],
                    text.len(),
                ) {
                    dfa::Result::Match(_) => Some(text.len()),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.shortest_nfa(text, start),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.shortest_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match(e) => Some(e),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.shortest_nfa(text, start),
                }
            }
            MatchType::Nfa(ty) => self.shortest_nfa_type(ty, text, start),
            MatchType::Nothing => None,
        }
    }

    /// Returns true if and only if the regex matches text.
    ///
    /// For single regular expressions, this is equivalent to calling
    /// shortest_match(...).is_some().
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_match_at(&self, text: &[u8], start: usize) -> bool {
        if !self.is_anchor_end_match(text) {
            return false;
        }
        // We need to do this dance because shortest_match relies on the NFA
        // filling in captures[1], but a RegexSet has no captures. In other
        // words, a RegexSet can't (currently) use shortest_match. ---AG
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => {
                self.find_literals(ty, text, start).is_some()
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa | MatchType::DfaMany => {
                match self.shortest_dfa(text, start) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.match_nfa(text, start),
                }
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match dfa::Fsm::reverse(
                    &self.ro.dfa_reverse,
                    self.cache.value(),
                    true,
                    &text[start..],
                    text.len(),
                ) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.match_nfa(text, start),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.shortest_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.match_nfa(text, start),
                }
            }
            MatchType::Nfa(ty) => self.match_nfa_type(ty, text, start),
            MatchType::Nothing => false,
        }
    }

    /// Finds the start and end location of the leftmost-first match, starting
    /// at the given location.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_at(&self, text: &[u8], start: usize) -> Option<(usize, usize)> {
        if !self.is_anchor_end_match(text) {
            return None;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => self.find_literals(ty, text, start),
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa => match self.find_dfa_forward(text, start) {
                dfa::Result::Match((s, e)) => Some((s, e)),
                dfa::Result::NoMatch(_) => None,
                dfa::Result::Quit => {
                    self.find_nfa(MatchNfaType::Auto, text, start)
                }
            },
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match self.find_dfa_anchored_reverse(text, start) {
                    dfa::Result::Match((s, e)) => Some((s, e)),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => {
                        self.find_nfa(MatchNfaType::Auto, text, start)
                    }
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.find_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match((s, e)) => Some((s, e)),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => {
                        self.find_nfa(MatchNfaType::Auto, text, start)
                    }
                }
            }
            MatchType::Nfa(ty) => self.find_nfa(ty, text, start),
            MatchType::Nothing => None,
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaMany => {
                unreachable!("BUG: RegexSet cannot be used with find")
            }
        }
    }

    /// Finds the start and end location of the leftmost-first match and also
    /// fills in all matching capture groups.
    ///
    /// The number of capture slots given should be equal to the total number
    /// of capture slots in the compiled program.
    ///
    /// Note that the first two slots always correspond to the start and end
    /// locations of the overall match.
    fn captures_read_at(
        &self,
        locs: &mut Locations,
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        let slots = locs.as_slots();
        for slot in slots.iter_mut() {
            *slot = None;
        }
        // If the caller unnecessarily uses this, then we try to save them
        // from themselves.
        match slots.len() {
            0 => return self.find_at(text, start),
            2 => {
                return self.find_at(text, start).map(|(s, e)| {
                    slots[0] = Some(s);
                    slots[1] = Some(e);
                    (s, e)
                });
            }
            _ => {} // fallthrough
        }
        if !self.is_anchor_end_match(text) {
            return None;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            MatchType::Literal(ty) => {
                self.find_literals(ty, text, start).and_then(|(s, e)| {
                    self.captures_nfa_type(
                        MatchNfaType::Auto,
                        slots,
                        text,
                        s,
                        e,
                    )
                })
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::Dfa => {
                if self.ro.nfa.is_anchored_start {
                    self.captures_nfa(slots, text, start)
                } else {
                    match self.find_dfa_forward(text, start) {
                        dfa::Result::Match((s, e)) => self.captures_nfa_type(
                            MatchNfaType::Auto,
                            slots,
                            text,
                            s,
                            e,
                        ),
                        dfa::Result::NoMatch(_) => None,
                        dfa::Result::Quit => {
                            self.captures_nfa(slots, text, start)
                        }
                    }
                }
            }
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaAnchoredReverse => {
                match self.find_dfa_anchored_reverse(text, start) {
                    dfa::Result::Match((s, e)) => self.captures_nfa_type(
                        MatchNfaType::Auto,
                        slots,
                        text,
                        s,
                        e,
                    ),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.captures_nfa(slots, text, start),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            MatchType::DfaSuffix => {
                match self.find_dfa_reverse_suffix(text, start) {
                    dfa::Result::Match((s, e)) => self.captures_nfa_type(
                        MatchNfaType::Auto,
                        slots,
                        text,
                        s,
                        e,
                    ),
                    dfa::Result::NoMatch(_) => None,
                    dfa::Result::Quit => self.captures_nfa(slots, text, start),
                }
            }
            MatchType::Nfa(ty) => {
                self.captures_nfa_type(ty, slots, text, start, text.len())
            }
            MatchType::Nothing => None,
            #[cfg(feature = "perf-dfa")]
            MatchType::DfaMany => {
                unreachable!("BUG: RegexSet cannot be used with captures")
            }
        }
    }
}

impl<'c> ExecNoSync<'c> {
    /// Finds the leftmost-first match using only literal search.
    #[cfg(feature = "perf-literal")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_literals(
        &self,
        ty: MatchLiteralType,
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        use self::MatchLiteralType::*;
        match ty {
            Unanchored => {
                let lits = &self.ro.nfa.prefixes;
                lits.find(&text[start..]).map(|(s, e)| (start + s, start + e))
            }
            AnchoredStart => {
                let lits = &self.ro.nfa.prefixes;
                if start == 0 || !self.ro.nfa.is_anchored_start {
                    lits.find_start(&text[start..])
                        .map(|(s, e)| (start + s, start + e))
                } else {
                    None
                }
            }
            AnchoredEnd => {
                let lits = &self.ro.suffixes;
                lits.find_end(&text[start..])
                    .map(|(s, e)| (start + s, start + e))
            }
            AhoCorasick => self
                .ro
                .ac
                .as_ref()
                .unwrap()
                .find(&text[start..])
                .map(|m| (start + m.start(), start + m.end())),
        }
    }

    /// Finds the leftmost-first match (start and end) using only the DFA.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(feature = "perf-dfa")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_dfa_forward(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<(usize, usize)> {
        use dfa::Result::*;
        let end = match dfa::Fsm::forward(
            &self.ro.dfa,
            self.cache.value(),
            false,
            text,
            start,
        ) {
            NoMatch(i) => return NoMatch(i),
            Quit => return Quit,
            Match(end) if start == end => return Match((start, start)),
            Match(end) => end,
        };
        // Now run the DFA in reverse to find the start of the match.
        match dfa::Fsm::reverse(
            &self.ro.dfa_reverse,
            self.cache.value(),
            false,
            &text[start..],
            end - start,
        ) {
            Match(s) => Match((start + s, end)),
            NoMatch(i) => NoMatch(i),
            Quit => Quit,
        }
    }

    /// Finds the leftmost-first match (start and end) using only the DFA,
    /// but assumes the regex is anchored at the end and therefore starts at
    /// the end of the regex and matches in reverse.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(feature = "perf-dfa")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_dfa_anchored_reverse(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<(usize, usize)> {
        use dfa::Result::*;
        match dfa::Fsm::reverse(
            &self.ro.dfa_reverse,
            self.cache.value(),
            false,
            &text[start..],
            text.len() - start,
        ) {
            Match(s) => Match((start + s, text.len())),
            NoMatch(i) => NoMatch(i),
            Quit => Quit,
        }
    }

    /// Finds the end of the shortest match using only the DFA.
    #[cfg(feature = "perf-dfa")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_dfa(&self, text: &[u8], start: usize) -> dfa::Result<usize> {
        dfa::Fsm::forward(&self.ro.dfa, self.cache.value(), true, text, start)
    }

    /// Finds the end of the shortest match using only the DFA by scanning for
    /// suffix literals.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn shortest_dfa_reverse_suffix(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<usize> {
        match self.exec_dfa_reverse_suffix(text, start) {
            None => self.shortest_dfa(text, start),
            Some(r) => r.map(|(_, end)| end),
        }
    }

    /// Finds the end of the shortest match using only the DFA by scanning for
    /// suffix literals. It also reports the start of the match.
    ///
    /// Note that if None is returned, then the optimization gave up to avoid
    /// worst case quadratic behavior. A forward scanning DFA should be tried
    /// next.
    ///
    /// If a match is returned and the full leftmost-first match is desired,
    /// then a forward scan starting from the beginning of the match must be
    /// done.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn exec_dfa_reverse_suffix(
        &self,
        text: &[u8],
        original_start: usize,
    ) -> Option<dfa::Result<(usize, usize)>> {
        use dfa::Result::*;

        let lcs = self.ro.suffixes.lcs();
        debug_assert!(lcs.len() >= 1);
        let mut start = original_start;
        let mut end = start;
        let mut last_literal = start;
        while end <= text.len() {
            last_literal += match lcs.find(&text[last_literal..]) {
                None => return Some(NoMatch(text.len())),
                Some(i) => i,
            };
            end = last_literal + lcs.len();
            match dfa::Fsm::reverse(
                &self.ro.dfa_reverse,
                self.cache.value(),
                false,
                &text[start..end],
                end - start,
            ) {
                Match(0) | NoMatch(0) => return None,
                Match(i) => return Some(Match((start + i, end))),
                NoMatch(i) => {
                    start += i;
                    last_literal += 1;
                    continue;
                }
                Quit => return Some(Quit),
            };
        }
        Some(NoMatch(text.len()))
    }

    /// Finds the leftmost-first match (start and end) using only the DFA
    /// by scanning for suffix literals.
    ///
    /// If the result returned indicates that the DFA quit, then another
    /// matching engine should be used.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find_dfa_reverse_suffix(
        &self,
        text: &[u8],
        start: usize,
    ) -> dfa::Result<(usize, usize)> {
        use dfa::Result::*;

        let match_start = match self.exec_dfa_reverse_suffix(text, start) {
            None => return self.find_dfa_forward(text, start),
            Some(Match((start, _))) => start,
            Some(r) => return r,
        };
        // At this point, we've found a match. The only way to quit now
        // without a match is if the DFA gives up (seems unlikely).
        //
        // Now run the DFA forwards to find the proper end of the match.
        // (The suffix literal match can only indicate the earliest
        // possible end location, which may appear before the end of the
        // leftmost-first match.)
        match dfa::Fsm::forward(
            &self.ro.dfa,
            self.cache.value(),
            false,
            text,
            match_start,
        ) {
            NoMatch(_) => panic!("BUG: reverse match implies forward match"),
            Quit => Quit,
            Match(e) => Match((match_start, e)),
        }
    }

    /// Executes the NFA engine to return whether there is a match or not.
    ///
    /// Ideally, we could use shortest_nfa(...).is_some() and get the same
    /// performance characteristics, but regex sets don't have captures, which
    /// shortest_nfa depends on.
    #[cfg(feature = "perf-dfa")]
    fn match_nfa(&self, text: &[u8], start: usize) -> bool {
        self.match_nfa_type(MatchNfaType::Auto, text, start)
    }

    /// Like match_nfa, but allows specification of the type of NFA engine.
    fn match_nfa_type(
        &self,
        ty: MatchNfaType,
        text: &[u8],
        start: usize,
    ) -> bool {
        self.exec_nfa(
            ty,
            &mut [false],
            &mut [],
            true,
            false,
            text,
            start,
            text.len(),
        )
    }

    /// Finds the shortest match using an NFA.
    #[cfg(feature = "perf-dfa")]
    fn shortest_nfa(&self, text: &[u8], start: usize) -> Option<usize> {
        self.shortest_nfa_type(MatchNfaType::Auto, text, start)
    }

    /// Like shortest_nfa, but allows specification of the type of NFA engine.
    fn shortest_nfa_type(
        &self,
        ty: MatchNfaType,
        text: &[u8],
        start: usize,
    ) -> Option<usize> {
        let mut slots = [None, None];
        if self.exec_nfa(
            ty,
            &mut [false],
            &mut slots,
            true,
            true,
            text,
            start,
            text.len(),
        ) {
            slots[1]
        } else {
            None
        }
    }

    /// Like find, but executes an NFA engine.
    fn find_nfa(
        &self,
        ty: MatchNfaType,
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        let mut slots = [None, None];
        if self.exec_nfa(
            ty,
            &mut [false],
            &mut slots,
            false,
            false,
            text,
            start,
            text.len(),
        ) {
            match (slots[0], slots[1]) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Like find_nfa, but fills in captures.
    ///
    /// `slots` should have length equal to `2 * nfa.captures.len()`.
    #[cfg(feature = "perf-dfa")]
    fn captures_nfa(
        &self,
        slots: &mut [Slot],
        text: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        self.captures_nfa_type(
            MatchNfaType::Auto,
            slots,
            text,
            start,
            text.len(),
        )
    }

    /// Like captures_nfa, but allows specification of type of NFA engine.
    fn captures_nfa_type(
        &self,
        ty: MatchNfaType,
        slots: &mut [Slot],
        text: &[u8],
        start: usize,
        end: usize,
    ) -> Option<(usize, usize)> {
        if self.exec_nfa(
            ty,
            &mut [false],
            slots,
            false,
            false,
            text,
            start,
            end,
        ) {
            match (slots[0], slots[1]) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            }
        } else {
            None
        }
    }

    fn exec_nfa(
        &self,
        mut ty: MatchNfaType,
        matches: &mut [bool],
        slots: &mut [Slot],
        quit_after_match: bool,
        quit_after_match_with_pos: bool,
        text: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        use self::MatchNfaType::*;
        if let Auto = ty {
            if backtrack::should_exec(self.ro.nfa.len(), text.len()) {
                ty = Backtrack;
            } else {
                ty = PikeVM;
            }
        }
        // The backtracker can't return the shortest match position as it is
        // implemented today. So if someone calls `shortest_match` and we need
        // to run an NFA, then use the PikeVM.
        if quit_after_match_with_pos || ty == PikeVM {
            self.exec_pikevm(
                matches,
                slots,
                quit_after_match,
                text,
                start,
                end,
            )
        } else {
            self.exec_backtrack(matches, slots, text, start, end)
        }
    }

    /// Always run the NFA algorithm.
    fn exec_pikevm(
        &self,
        matches: &mut [bool],
        slots: &mut [Slot],
        quit_after_match: bool,
        text: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        if self.ro.nfa.uses_bytes() {
            pikevm::Fsm::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                quit_after_match,
                ByteInput::new(text, self.ro.nfa.only_utf8),
                start,
                end,
            )
        } else {
            pikevm::Fsm::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                quit_after_match,
                CharInput::new(text),
                start,
                end,
            )
        }
    }

    /// Always runs the NFA using bounded backtracking.
    fn exec_backtrack(
        &self,
        matches: &mut [bool],
        slots: &mut [Slot],
        text: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        if self.ro.nfa.uses_bytes() {
            backtrack::Bounded::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                ByteInput::new(text, self.ro.nfa.only_utf8),
                start,
                end,
            )
        } else {
            backtrack::Bounded::exec(
                &self.ro.nfa,
                self.cache.value(),
                matches,
                slots,
                CharInput::new(text),
                start,
                end,
            )
        }
    }

    /// Finds which regular expressions match the given text.
    ///
    /// `matches` should have length equal to the number of regexes being
    /// searched.
    ///
    /// This is only useful when one wants to know which regexes in a set
    /// match some text.
    pub fn many_matches_at(
        &self,
        matches: &mut [bool],
        text: &[u8],
        start: usize,
    ) -> bool {
        use self::MatchType::*;
        if !self.is_anchor_end_match(text) {
            return false;
        }
        match self.ro.match_type {
            #[cfg(feature = "perf-literal")]
            Literal(ty) => {
                debug_assert_eq!(matches.len(), 1);
                matches[0] = self.find_literals(ty, text, start).is_some();
                matches[0]
            }
            #[cfg(feature = "perf-dfa")]
            Dfa | DfaAnchoredReverse | DfaMany => {
                match dfa::Fsm::forward_many(
                    &self.ro.dfa,
                    self.cache.value(),
                    matches,
                    text,
                    start,
                ) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.exec_nfa(
                        MatchNfaType::Auto,
                        matches,
                        &mut [],
                        false,
                        false,
                        text,
                        start,
                        text.len(),
                    ),
                }
            }
            #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
            DfaSuffix => {
                match dfa::Fsm::forward_many(
                    &self.ro.dfa,
                    self.cache.value(),
                    matches,
                    text,
                    start,
                ) {
                    dfa::Result::Match(_) => true,
                    dfa::Result::NoMatch(_) => false,
                    dfa::Result::Quit => self.exec_nfa(
                        MatchNfaType::Auto,
                        matches,
                        &mut [],
                        false,
                        false,
                        text,
                        start,
                        text.len(),
                    ),
                }
            }
            Nfa(ty) => self.exec_nfa(
                ty,
                matches,
                &mut [],
                false,
                false,
                text,
                start,
                text.len(),
            ),
            Nothing => false,
        }
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_anchor_end_match(&self, text: &[u8]) -> bool {
        #[cfg(not(feature = "perf-literal"))]
        fn imp(_: &ExecReadOnly, _: &[u8]) -> bool {
            true
        }

        #[cfg(feature = "perf-literal")]
        fn imp(ro: &ExecReadOnly, text: &[u8]) -> bool {
            // Only do this check if the haystack is big (>1MB).
            if text.len() > (1 << 20) && ro.nfa.is_anchored_end {
                let lcs = ro.suffixes.lcs();
                if lcs.len() >= 1 && !lcs.is_suffix(text) {
                    return false;
                }
            }
            true
        }

        imp(&self.ro, text)
    }

    pub fn capture_name_idx(&self) -> &Arc<HashMap<String, usize>> {
        &self.ro.nfa.capture_name_idx
    }
}

impl<'c> ExecNoSyncStr<'c> {
    pub fn capture_name_idx(&self) -> &Arc<HashMap<String, usize>> {
        self.0.capture_name_idx()
    }
}

impl Exec {
    /// Get a searcher that isn't Sync.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn searcher(&self) -> ExecNoSync {
        let create = || RefCell::new(ProgramCacheInner::new(&self.ro));
        ExecNoSync {
            ro: &self.ro, // a clone is too expensive here! (and not needed)
            cache: self.cache.get_or(create),
        }
    }

    /// Get a searcher that isn't Sync and can match on &str.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn searcher_str(&self) -> ExecNoSyncStr {
        ExecNoSyncStr(self.searcher())
    }

    /// Build a Regex from this executor.
    pub fn into_regex(self) -> re_unicode::Regex {
        re_unicode::Regex::from(self)
    }

    /// Build a RegexSet from this executor.
    pub fn into_regex_set(self) -> re_set::unicode::RegexSet {
        re_set::unicode::RegexSet::from(self)
    }

    /// Build a Regex from this executor that can match arbitrary bytes.
    pub fn into_byte_regex(self) -> re_bytes::Regex {
        re_bytes::Regex::from(self)
    }

    /// Build a RegexSet from this executor that can match arbitrary bytes.
    pub fn into_byte_regex_set(self) -> re_set::bytes::RegexSet {
        re_set::bytes::RegexSet::from(self)
    }

    /// The original regular expressions given by the caller that were
    /// compiled.
    pub fn regex_strings(&self) -> &[String] {
        &self.ro.res
    }

    /// Return a slice of capture names.
    ///
    /// Any capture that isn't named is None.
    pub fn capture_names(&self) -> &[Option<String>] {
        &self.ro.nfa.captures
    }

    /// Return a reference to named groups mapping (from group name to
    /// group position).
    pub fn capture_name_idx(&self) -> &Arc<HashMap<String, usize>> {
        &self.ro.nfa.capture_name_idx
    }
}

impl Clone for Exec {
    fn clone(&self) -> Exec {
        Exec { ro: self.ro.clone(), cache: Cached::new() }
    }
}

impl ExecReadOnly {
    fn choose_match_type(&self, hint: Option<MatchType>) -> MatchType {
        if let Some(MatchType::Nfa(_)) = hint {
            return hint.unwrap();
        }
        // If the NFA is empty, then we'll never match anything.
        if self.nfa.insts.is_empty() {
            return MatchType::Nothing;
        }
        if let Some(literalty) = self.choose_literal_match_type() {
            return literalty;
        }
        if let Some(dfaty) = self.choose_dfa_match_type() {
            return dfaty;
        }
        // We're so totally hosed.
        MatchType::Nfa(MatchNfaType::Auto)
    }

    /// If a plain literal scan can be used, then a corresponding literal
    /// search type is returned.
    fn choose_literal_match_type(&self) -> Option<MatchType> {
        #[cfg(not(feature = "perf-literal"))]
        fn imp(_: &ExecReadOnly) -> Option<MatchType> {
            None
        }

        #[cfg(feature = "perf-literal")]
        fn imp(ro: &ExecReadOnly) -> Option<MatchType> {
            // If our set of prefixes is complete, then we can use it to find
            // a match in lieu of a regex engine. This doesn't quite work well
            // in the presence of multiple regexes, so only do it when there's
            // one.
            //
            // TODO(burntsushi): Also, don't try to match literals if the regex
            // is partially anchored. We could technically do it, but we'd need
            // to create two sets of literals: all of them and then the subset
            // that aren't anchored. We would then only search for all of them
            // when at the beginning of the input and use the subset in all
            // other cases.
            if ro.res.len() != 1 {
                return None;
            }
            if ro.ac.is_some() {
                return Some(MatchType::Literal(
                    MatchLiteralType::AhoCorasick,
                ));
            }
            if ro.nfa.prefixes.complete() {
                return if ro.nfa.is_anchored_start {
                    Some(MatchType::Literal(MatchLiteralType::AnchoredStart))
                } else {
                    Some(MatchType::Literal(MatchLiteralType::Unanchored))
                };
            }
            if ro.suffixes.complete() {
                return if ro.nfa.is_anchored_end {
                    Some(MatchType::Literal(MatchLiteralType::AnchoredEnd))
                } else {
                    // This case shouldn't happen. When the regex isn't
                    // anchored, then complete prefixes should imply complete
                    // suffixes.
                    Some(MatchType::Literal(MatchLiteralType::Unanchored))
                };
            }
            None
        }

        imp(self)
    }

    /// If a DFA scan can be used, then choose the appropriate DFA strategy.
    fn choose_dfa_match_type(&self) -> Option<MatchType> {
        #[cfg(not(feature = "perf-dfa"))]
        fn imp(_: &ExecReadOnly) -> Option<MatchType> {
            None
        }

        #[cfg(feature = "perf-dfa")]
        fn imp(ro: &ExecReadOnly) -> Option<MatchType> {
            if !dfa::can_exec(&ro.dfa) {
                return None;
            }
            // Regex sets require a slightly specialized path.
            if ro.res.len() >= 2 {
                return Some(MatchType::DfaMany);
            }
            // If the regex is anchored at the end but not the start, then
            // just match in reverse from the end of the haystack.
            if !ro.nfa.is_anchored_start && ro.nfa.is_anchored_end {
                return Some(MatchType::DfaAnchoredReverse);
            }
            #[cfg(feature = "perf-literal")]
            {
                // If there's a longish suffix literal, then it might be faster
                // to look for that first.
                if ro.should_suffix_scan() {
                    return Some(MatchType::DfaSuffix);
                }
            }
            // Fall back to your garden variety forward searching lazy DFA.
            Some(MatchType::Dfa)
        }

        imp(self)
    }

    /// Returns true if the program is amenable to suffix scanning.
    ///
    /// When this is true, as a heuristic, we assume it is OK to quickly scan
    /// for suffix literals and then do a *reverse* DFA match from any matches
    /// produced by the literal scan. (And then followed by a forward DFA
    /// search, since the previously found suffix literal maybe not actually be
    /// the end of a match.)
    ///
    /// This is a bit of a specialized optimization, but can result in pretty
    /// big performance wins if 1) there are no prefix literals and 2) the
    /// suffix literals are pretty rare in the text. (1) is obviously easy to
    /// account for but (2) is harder. As a proxy, we assume that longer
    /// strings are generally rarer, so we only enable this optimization when
    /// we have a meaty suffix.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    fn should_suffix_scan(&self) -> bool {
        if self.suffixes.is_empty() {
            return false;
        }
        let lcs_len = self.suffixes.lcs().char_len();
        lcs_len >= 3 && lcs_len > self.dfa.prefixes.lcp().char_len()
    }
}

#[derive(Clone, Copy, Debug)]
enum MatchType {
    /// A single or multiple literal search. This is only used when the regex
    /// can be decomposed into a literal search.
    #[cfg(feature = "perf-literal")]
    Literal(MatchLiteralType),
    /// A normal DFA search.
    #[cfg(feature = "perf-dfa")]
    Dfa,
    /// A reverse DFA search starting from the end of a haystack.
    #[cfg(feature = "perf-dfa")]
    DfaAnchoredReverse,
    /// A reverse DFA search with suffix literal scanning.
    #[cfg(all(feature = "perf-dfa", feature = "perf-literal"))]
    DfaSuffix,
    /// Use the DFA on two or more regular expressions.
    #[cfg(feature = "perf-dfa")]
    DfaMany,
    /// An NFA variant.
    Nfa(MatchNfaType),
    /// No match is ever possible, so don't ever try to search.
    Nothing,
}

#[derive(Clone, Copy, Debug)]
#[cfg(feature = "perf-literal")]
enum MatchLiteralType {
    /// Match literals anywhere in text.
    Unanchored,
    /// Match literals only at the start of text.
    AnchoredStart,
    /// Match literals only at the end of text.
    AnchoredEnd,
    /// Use an Aho-Corasick automaton. This requires `ac` to be Some on
    /// ExecReadOnly.
    AhoCorasick,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatchNfaType {
    /// Choose between Backtrack and PikeVM.
    Auto,
    /// NFA bounded backtracking.
    ///
    /// (This is only set by tests, since it never makes sense to always want
    /// backtracking.)
    Backtrack,
    /// The Pike VM.
    ///
    /// (This is only set by tests, since it never makes sense to always want
    /// the Pike VM.)
    PikeVM,
}

/// `ProgramCache` maintains reusable allocations for each matching engine
/// available to a particular program.
pub type ProgramCache = RefCell<ProgramCacheInner>;

#[derive(Debug)]
pub struct ProgramCacheInner {
    pub pikevm: pikevm::Cache,
    pub backtrack: backtrack::Cache,
    #[cfg(feature = "perf-dfa")]
    pub dfa: dfa::Cache,
    #[cfg(feature = "perf-dfa")]
    pub dfa_reverse: dfa::Cache,
}

impl ProgramCacheInner {
    fn new(ro: &ExecReadOnly) -> Self {
        ProgramCacheInner {
            pikevm: pikevm::Cache::new(&ro.nfa),
            backtrack: backtrack::Cache::new(&ro.nfa),
            #[cfg(feature = "perf-dfa")]
            dfa: dfa::Cache::new(&ro.dfa),
            #[cfg(feature = "perf-dfa")]
            dfa_reverse: dfa::Cache::new(&ro.dfa_reverse),
        }
    }
}

/// Alternation literals checks if the given HIR is a simple alternation of
/// literals, and if so, returns them. Otherwise, this returns None.
#[cfg(feature = "perf-literal")]
fn alternation_literals(expr: &Hir) -> Option<Vec<Vec<u8>>> {
    use syntax::hir::{HirKind, Literal};

    // This is pretty hacky, but basically, if `is_alternation_literal` is
    // true, then we can make several assumptions about the structure of our
    // HIR. This is what justifies the `unreachable!` statements below.
    //
    // This code should be refactored once we overhaul this crate's
    // optimization pipeline, because this is a terribly inflexible way to go
    // about things.

    if !expr.is_alternation_literal() {
        return None;
    }
    let alts = match *expr.kind() {
        HirKind::Alternation(ref alts) => alts,
        _ => return None, // one literal isn't worth it
    };

    let extendlit = |lit: &Literal, dst: &mut Vec<u8>| match *lit {
        Literal::Unicode(c) => {
            let mut buf = [0; 4];
            dst.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
        Literal::Byte(b) => {
            dst.push(b);
        }
    };

    let mut lits = vec![];
    for alt in alts {
        let mut lit = vec![];
        match *alt.kind() {
            HirKind::Literal(ref x) => extendlit(x, &mut lit),
            HirKind::Concat(ref exprs) => {
                for e in exprs {
                    match *e.kind() {
                        HirKind::Literal(ref x) => extendlit(x, &mut lit),
                        _ => unreachable!("expected literal, got {:?}", e),
                    }
                }
            }
            _ => unreachable!("expected literal or concat, got {:?}", alt),
        }
        lits.push(lit);
    }
    Some(lits)
}

#[cfg(test)]
mod test {
    #[test]
    fn uppercut_s_backtracking_bytes_default_bytes_mismatch() {
        use internal::ExecBuilder;

        let backtrack_bytes_re = ExecBuilder::new("^S")
            .bounded_backtracking()
            .only_utf8(false)
            .build()
            .map(|exec| exec.into_byte_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let default_bytes_re = ExecBuilder::new("^S")
            .only_utf8(false)
            .build()
            .map(|exec| exec.into_byte_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let input = vec![83, 83];

        let s1 = backtrack_bytes_re.split(&input);
        let s2 = default_bytes_re.split(&input);
        for (chunk1, chunk2) in s1.zip(s2) {
            assert_eq!(chunk1, chunk2);
        }
    }

    #[test]
    fn unicode_lit_star_backtracking_utf8bytes_default_utf8bytes_mismatch() {
        use internal::ExecBuilder;

        let backtrack_bytes_re = ExecBuilder::new(r"^(?u:\*)")
            .bounded_backtracking()
            .bytes(true)
            .build()
            .map(|exec| exec.into_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let default_bytes_re = ExecBuilder::new(r"^(?u:\*)")
            .bytes(true)
            .build()
            .map(|exec| exec.into_regex())
            .map_err(|err| format!("{}", err))
            .unwrap();

        let input = "**";

        let s1 = backtrack_bytes_re.split(input);
        let s2 = default_bytes_re.split(input);
        for (chunk1, chunk2) in s1.zip(s2) {
            assert_eq!(chunk1, chunk2);
        }
    }
}
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::slice;
use std::sync::Arc;

use input::Char;
use literal::LiteralSearcher;

/// `InstPtr` represents the index of an instruction in a regex program.
pub type InstPtr = usize;

/// Program is a sequence of instructions and various facts about thos
/// instructions.
#[derive(Clone)]
pub struct Program {
    /// A sequence of instructions that represents an NFA.
    pub insts: Vec<Inst>,
    /// Pointers to each Match instruction in the sequence.
    ///
    /// This is always length 1 unless this program represents a regex set.
    pub matches: Vec<InstPtr>,
    /// The ordered sequence of all capture groups extracted from the AST.
    /// Unnamed groups are `None`.
    pub captures: Vec<Option<String>>,
    /// Pointers to all named capture groups into `captures`.
    pub capture_name_idx: Arc<HashMap<String, usize>>,
    /// A pointer to the start instruction. This can vary depending on how
    /// the program was compiled. For example, programs for use with the DFA
    /// engine have a `.*?` inserted at the beginning of unanchored regular
    /// expressions. The actual starting point of the program is after the
    /// `.*?`.
    pub start: InstPtr,
    /// A set of equivalence classes for discriminating bytes in the compiled
    /// program.
    pub byte_classes: Vec<u8>,
    /// When true, this program can only match valid UTF-8.
    pub only_utf8: bool,
    /// When true, this program uses byte range instructions instead of Unicode
    /// range instructions.
    pub is_bytes: bool,
    /// When true, the program is compiled for DFA matching. For example, this
    /// implies `is_bytes` and also inserts a preceding `.*?` for unanchored
    /// regexes.
    pub is_dfa: bool,
    /// When true, the program matches text in reverse (for use only in the
    /// DFA).
    pub is_reverse: bool,
    /// Whether the regex must match from the start of the input.
    pub is_anchored_start: bool,
    /// Whether the regex must match at the end of the input.
    pub is_anchored_end: bool,
    /// Whether this program contains a Unicode word boundary instruction.
    pub has_unicode_word_boundary: bool,
    /// A possibly empty machine for very quickly matching prefix literals.
    pub prefixes: LiteralSearcher,
    /// A limit on the size of the cache that the DFA is allowed to use while
    /// matching.
    ///
    /// The cache limit specifies approximately how much space we're willing to
    /// give to the state cache. Once the state cache exceeds the size, it is
    /// wiped and all states must be re-computed.
    ///
    /// Note that this value does not impact correctness. It can be set to 0
    /// and the DFA will run just fine. (It will only ever store exactly one
    /// state in the cache, and will likely run very slowly, but it will work.)
    ///
    /// Also note that this limit is *per thread of execution*. That is,
    /// if the same regex is used to search text across multiple threads
    /// simultaneously, then the DFA cache is not shared. Instead, copies are
    /// made.
    pub dfa_size_limit: usize,
}

impl Program {
    /// Creates an empty instruction sequence. Fields are given default
    /// values.
    pub fn new() -> Self {
        Program {
            insts: vec![],
            matches: vec![],
            captures: vec![],
            capture_name_idx: Arc::new(HashMap::new()),
            start: 0,
            byte_classes: vec![0; 256],
            only_utf8: true,
            is_bytes: false,
            is_dfa: false,
            is_reverse: false,
            is_anchored_start: false,
            is_anchored_end: false,
            has_unicode_word_boundary: false,
            prefixes: LiteralSearcher::empty(),
            dfa_size_limit: 2 * (1 << 20),
        }
    }

    /// If pc is an index to a no-op instruction (like Save), then return the
    /// next pc that is not a no-op instruction.
    pub fn skip(&self, mut pc: usize) -> usize {
        loop {
            match self[pc] {
                Inst::Save(ref i) => pc = i.goto,
                _ => return pc,
            }
        }
    }

    /// Return true if and only if an execution engine at instruction `pc` will
    /// always lead to a match.
    pub fn leads_to_match(&self, pc: usize) -> bool {
        if self.matches.len() > 1 {
            // If we have a regex set, then we have more than one ending
            // state, so leading to one of those states is generally
            // meaningless.
            return false;
        }
        match self[self.skip(pc)] {
            Inst::Match(_) => true,
            _ => false,
        }
    }

    /// Returns true if the current configuration demands that an implicit
    /// `.*?` be prepended to the instruction sequence.
    pub fn needs_dotstar(&self) -> bool {
        self.is_dfa && !self.is_reverse && !self.is_anchored_start
    }

    /// Returns true if this program uses Byte instructions instead of
    /// Char/Range instructions.
    pub fn uses_bytes(&self) -> bool {
        self.is_bytes || self.is_dfa
    }

    /// Returns true if this program exclusively matches valid UTF-8 bytes.
    ///
    /// That is, if an invalid UTF-8 byte is seen, then no match is possible.
    pub fn only_utf8(&self) -> bool {
        self.only_utf8
    }

    /// Return the approximate heap usage of this instruction sequence in
    /// bytes.
    pub fn approximate_size(&self) -> usize {
        // The only instruction that uses heap space is Ranges (for
        // Unicode codepoint programs) to store non-overlapping codepoint
        // ranges. To keep this operation constant time, we ignore them.
        (self.len() * mem::size_of::<Inst>())
            + (self.matches.len() * mem::size_of::<InstPtr>())
            + (self.captures.len() * mem::size_of::<Option<String>>())
            + (self.capture_name_idx.len()
                * (mem::size_of::<String>() + mem::size_of::<usize>()))
            + (self.byte_classes.len() * mem::size_of::<u8>())
            + self.prefixes.approximate_size()
    }
}

impl Deref for Program {
    type Target = [Inst];

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn deref(&self) -> &Self::Target {
        &*self.insts
    }
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Inst::*;

        fn with_goto(cur: usize, goto: usize, fmtd: String) -> String {
            if goto == cur + 1 {
                fmtd
            } else {
                format!("{} (goto: {})", fmtd, goto)
            }
        }

        fn visible_byte(b: u8) -> String {
            use std::ascii::escape_default;
            let escaped = escape_default(b).collect::<Vec<u8>>();
            String::from_utf8_lossy(&escaped).into_owned()
        }

        for (pc, inst) in self.iter().enumerate() {
            match *inst {
                Match(slot) => write!(f, "{:04} Match({:?})", pc, slot)?,
                Save(ref inst) => {
                    let s = format!("{:04} Save({})", pc, inst.slot);
                    write!(f, "{}", with_goto(pc, inst.goto, s))?;
                }
                Split(ref inst) => {
                    write!(
                        f,
                        "{:04} Split({}, {})",
                        pc, inst.goto1, inst.goto2
                    )?;
                }
                EmptyLook(ref inst) => {
                    let s = format!("{:?}", inst.look);
                    write!(f, "{:04} {}", pc, with_goto(pc, inst.goto, s))?;
                }
                Char(ref inst) => {
                    let s = format!("{:?}", inst.c);
                    write!(f, "{:04} {}", pc, with_goto(pc, inst.goto, s))?;
                }
                Ranges(ref inst) => {
                    let ranges = inst
                        .ranges
                        .iter()
                        .map(|r| format!("{:?}-{:?}", r.0, r.1))
                        .collect::<Vec<String>>()
                        .join(", ");
                    write!(
                        f,
                        "{:04} {}",
                        pc,
                        with_goto(pc, inst.goto, ranges)
                    )?;
                }
                Bytes(ref inst) => {
                    let s = format!(
                        "Bytes({}, {})",
                        visible_byte(inst.start),
                        visible_byte(inst.end)
                    );
                    write!(f, "{:04} {}", pc, with_goto(pc, inst.goto, s))?;
                }
            }
            if pc == self.start {
                write!(f, " (start)")?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a Program {
    type Item = &'a Inst;
    type IntoIter = slice::Iter<'a, Inst>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Inst is an instruction code in a Regex program.
///
/// Regrettably, a regex program either contains Unicode codepoint
/// instructions (Char and Ranges) or it contains byte instructions (Bytes).
/// A regex program can never contain both.
///
/// It would be worth investigating splitting this into two distinct types and
/// then figuring out how to make the matching engines polymorphic over those
/// types without sacrificing performance.
///
/// Other than the benefit of moving invariants into the type system, another
/// benefit is the decreased size. If we remove the `Char` and `Ranges`
/// instructions from the `Inst` enum, then its size shrinks from 40 bytes to
/// 24 bytes. (This is because of the removal of a `Vec` in the `Ranges`
/// variant.) Given that byte based machines are typically much bigger than
/// their Unicode analogues (because they can decode UTF-8 directly), this ends
/// up being a pretty significant savings.
#[derive(Clone, Debug)]
pub enum Inst {
    /// Match indicates that the program has reached a match state.
    ///
    /// The number in the match corresponds to the Nth logical regular
    /// expression in this program. This index is always 0 for normal regex
    /// programs. Values greater than 0 appear when compiling regex sets, and
    /// each match instruction gets its own unique value. The value corresponds
    /// to the Nth regex in the set.
    Match(usize),
    /// Save causes the program to save the current location of the input in
    /// the slot indicated by InstSave.
    Save(InstSave),
    /// Split causes the program to diverge to one of two paths in the
    /// program, preferring goto1 in InstSplit.
    Split(InstSplit),
    /// EmptyLook represents a zero-width assertion in a regex program. A
    /// zero-width assertion does not consume any of the input text.
    EmptyLook(InstEmptyLook),
    /// Char requires the regex program to match the character in InstChar at
    /// the current position in the input.
    Char(InstChar),
    /// Ranges requires the regex program to match the character at the current
    /// position in the input with one of the ranges specified in InstRanges.
    Ranges(InstRanges),
    /// Bytes is like Ranges, except it expresses a single byte range. It is
    /// used in conjunction with Split instructions to implement multi-byte
    /// character classes.
    Bytes(InstBytes),
}

impl Inst {
    /// Returns true if and only if this is a match instruction.
    pub fn is_match(&self) -> bool {
        match *self {
            Inst::Match(_) => true,
            _ => false,
        }
    }
}

/// Representation of the Save instruction.
#[derive(Clone, Debug)]
pub struct InstSave {
    /// The next location to execute in the program.
    pub goto: InstPtr,
    /// The capture slot (there are two slots for every capture in a regex,
    /// including the zeroth capture for the entire match).
    pub slot: usize,
}

/// Representation of the Split instruction.
#[derive(Clone, Debug)]
pub struct InstSplit {
    /// The first instruction to try. A match resulting from following goto1
    /// has precedence over a match resulting from following goto2.
    pub goto1: InstPtr,
    /// The second instruction to try. A match resulting from following goto1
    /// has precedence over a match resulting from following goto2.
    pub goto2: InstPtr,
}

/// Representation of the `EmptyLook` instruction.
#[derive(Clone, Debug)]
pub struct InstEmptyLook {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The type of zero-width assertion to check.
    pub look: EmptyLook,
}

/// The set of zero-width match instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyLook {
    /// Start of line or input.
    StartLine,
    /// End of line or input.
    EndLine,
    /// Start of input.
    StartText,
    /// End of input.
    EndText,
    /// Word character on one side and non-word character on other.
    WordBoundary,
    /// Word character on both sides or non-word character on both sides.
    NotWordBoundary,
    /// ASCII word boundary.
    WordBoundaryAscii,
    /// Not ASCII word boundary.
    NotWordBoundaryAscii,
}

/// Representation of the Char instruction.
#[derive(Clone, Debug)]
pub struct InstChar {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The character to test.
    pub c: char,
}

/// Representation of the Ranges instruction.
#[derive(Clone, Debug)]
pub struct InstRanges {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The set of Unicode scalar value ranges to test.
    pub ranges: Vec<(char, char)>,
}

impl InstRanges {
    /// Tests whether the given input character matches this instruction.
    pub fn matches(&self, c: Char) -> bool {
        // This speeds up the `match_class_unicode` benchmark by checking
        // some common cases quickly without binary search. e.g., Matching
        // a Unicode class on predominantly ASCII text.
        for r in self.ranges.iter().take(4) {
            if c < r.0 {
                return false;
            }
            if c <= r.1 {
                return true;
            }
        }
        self.ranges
            .binary_search_by(|r| {
                if r.1 < c {
                    Ordering::Less
                } else if r.0 > c {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok()
    }

    /// Return the number of distinct characters represented by all of the
    /// ranges.
    pub fn num_chars(&self) -> usize {
        self.ranges
            .iter()
            .map(|&(s, e)| 1 + (e as u32) - (s as u32))
            .sum::<u32>() as usize
    }
}

/// Representation of the Bytes instruction.
#[derive(Clone, Debug)]
pub struct InstBytes {
    /// The next location to execute in the program if this instruction
    /// succeeds.
    pub goto: InstPtr,
    /// The start (inclusive) of this byte range.
    pub start: u8,
    /// The end (inclusive) of this byte range.
    pub end: u8,
}

impl InstBytes {
    /// Returns true if and only if the given byte is in this range.
    pub fn matches(&self, byte: u8) -> bool {
        self.start <= byte && byte <= self.end
    }
}
/// A few elementary UTF-8 encoding and decoding functions used by the matching
/// engines.
///
/// In an ideal world, the matching engines operate on `&str` and we can just
/// lean on the standard library for all our UTF-8 needs. However, to support
/// byte based regexes (that can match on arbitrary bytes which may contain
/// UTF-8), we need to be capable of searching and decoding UTF-8 on a `&[u8]`.
/// The standard library doesn't really recognize this use case, so we have
/// to build it out ourselves.
///
/// Should this be factored out into a separate crate? It seems independently
/// useful. There are other crates that already exist (e.g., `utf-8`) that have
/// overlapping use cases. Not sure what to do.
use std::char;

const TAG_CONT: u8 = 0b1000_0000;
const TAG_TWO: u8 = 0b1100_0000;
const TAG_THREE: u8 = 0b1110_0000;
const TAG_FOUR: u8 = 0b1111_0000;

/// Returns the smallest possible index of the next valid UTF-8 sequence
/// starting after `i`.
pub fn next_utf8(text: &[u8], i: usize) -> usize {
    let b = match text.get(i) {
        None => return i + 1,
        Some(&b) => b,
    };
    let inc = if b <= 0x7F {
        1
    } else if b <= 0b110_11111 {
        2
    } else if b <= 0b1110_1111 {
        3
    } else {
        4
    };
    i + inc
}

/// Decode a single UTF-8 sequence into a single Unicode codepoint from `src`.
///
/// If no valid UTF-8 sequence could be found, then `None` is returned.
/// Otherwise, the decoded codepoint and the number of bytes read is returned.
/// The number of bytes read (for a valid UTF-8 sequence) is guaranteed to be
/// 1, 2, 3 or 4.
///
/// Note that a UTF-8 sequence is invalid if it is incorrect UTF-8, encodes a
/// codepoint that is out of range (surrogate codepoints are out of range) or
/// is not the shortest possible UTF-8 sequence for that codepoint.
#[inline]
pub fn decode_utf8(src: &[u8]) -> Option<(char, usize)> {
    let b0 = match src.get(0) {
        None => return None,
        Some(&b) if b <= 0x7F => return Some((b as char, 1)),
        Some(&b) => b,
    };
    match b0 {
        0b110_00000..=0b110_11111 => {
            if src.len() < 2 {
                return None;
            }
            let b1 = src[1];
            if 0b11_000000 & b1 != TAG_CONT {
                return None;
            }
            let cp = ((b0 & !TAG_TWO) as u32) << 6 | ((b1 & !TAG_CONT) as u32);
            match cp {
                0x80..=0x7FF => char::from_u32(cp).map(|cp| (cp, 2)),
                _ => None,
            }
        }
        0b1110_0000..=0b1110_1111 => {
            if src.len() < 3 {
                return None;
            }
            let (b1, b2) = (src[1], src[2]);
            if 0b11_000000 & b1 != TAG_CONT {
                return None;
            }
            if 0b11_000000 & b2 != TAG_CONT {
                return None;
            }
            let cp = ((b0 & !TAG_THREE) as u32) << 12
                | ((b1 & !TAG_CONT) as u32) << 6
                | ((b2 & !TAG_CONT) as u32);
            match cp {
                // char::from_u32 will disallow surrogate codepoints.
                0x800..=0xFFFF => char::from_u32(cp).map(|cp| (cp, 3)),
                _ => None,
            }
        }
        0b11110_000..=0b11110_111 => {
            if src.len() < 4 {
                return None;
            }
            let (b1, b2, b3) = (src[1], src[2], src[3]);
            if 0b11_000000 & b1 != TAG_CONT {
                return None;
            }
            if 0b11_000000 & b2 != TAG_CONT {
                return None;
            }
            if 0b11_000000 & b3 != TAG_CONT {
                return None;
            }
            let cp = ((b0 & !TAG_FOUR) as u32) << 18
                | ((b1 & !TAG_CONT) as u32) << 12
                | ((b2 & !TAG_CONT) as u32) << 6
                | ((b3 & !TAG_CONT) as u32);
            match cp {
                0x10000..=0x10FFFF => char::from_u32(cp).map(|cp| (cp, 4)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Like `decode_utf8`, but decodes the last UTF-8 sequence in `src` instead
/// of the first.
pub fn decode_last_utf8(src: &[u8]) -> Option<(char, usize)> {
    if src.is_empty() {
        return None;
    }
    let mut start = src.len() - 1;
    if src[start] <= 0x7F {
        return Some((src[start] as char, 1));
    }
    while start > src.len().saturating_sub(4) {
        start -= 1;
        if is_start_byte(src[start]) {
            break;
        }
    }
    match decode_utf8(&src[start..]) {
        None => None,
        Some((_, n)) if n < src.len() - start => None,
        Some((cp, n)) => Some((cp, n)),
    }
}

fn is_start_byte(b: u8) -> bool {
    b & 0b11_000000 != 0b1_0000000
}

#[cfg(test)]
mod tests {
    use std::str;

    use quickcheck::quickcheck;

    use super::{
        decode_last_utf8, decode_utf8, TAG_CONT, TAG_FOUR, TAG_THREE, TAG_TWO,
    };

    #[test]
    fn prop_roundtrip() {
        fn p(given_cp: char) -> bool {
            let mut tmp = [0; 4];
            let encoded_len = given_cp.encode_utf8(&mut tmp).len();
            let (got_cp, got_len) = decode_utf8(&tmp[..encoded_len]).unwrap();
            encoded_len == got_len && given_cp == got_cp
        }
        quickcheck(p as fn(char) -> bool)
    }

    #[test]
    fn prop_roundtrip_last() {
        fn p(given_cp: char) -> bool {
            let mut tmp = [0; 4];
            let encoded_len = given_cp.encode_utf8(&mut tmp).len();
            let (got_cp, got_len) =
                decode_last_utf8(&tmp[..encoded_len]).unwrap();
            encoded_len == got_len && given_cp == got_cp
        }
        quickcheck(p as fn(char) -> bool)
    }

    #[test]
    fn prop_encode_matches_std() {
        fn p(cp: char) -> bool {
            let mut got = [0; 4];
            let n = cp.encode_utf8(&mut got).len();
            let expected = cp.to_string();
            &got[..n] == expected.as_bytes()
        }
        quickcheck(p as fn(char) -> bool)
    }

    #[test]
    fn prop_decode_matches_std() {
        fn p(given_cp: char) -> bool {
            let mut tmp = [0; 4];
            let n = given_cp.encode_utf8(&mut tmp).len();
            let (got_cp, _) = decode_utf8(&tmp[..n]).unwrap();
            let expected_cp =
                str::from_utf8(&tmp[..n]).unwrap().chars().next().unwrap();
            got_cp == expected_cp
        }
        quickcheck(p as fn(char) -> bool)
    }

    #[test]
    fn prop_decode_last_matches_std() {
        fn p(given_cp: char) -> bool {
            let mut tmp = [0; 4];
            let n = given_cp.encode_utf8(&mut tmp).len();
            let (got_cp, _) = decode_last_utf8(&tmp[..n]).unwrap();
            let expected_cp = str::from_utf8(&tmp[..n])
                .unwrap()
                .chars()
                .rev()
                .next()
                .unwrap();
            got_cp == expected_cp
        }
        quickcheck(p as fn(char) -> bool)
    }

    #[test]
    fn reject_invalid() {
        // Invalid start byte
        assert_eq!(decode_utf8(&[0xFF]), None);
        // Surrogate pair
        assert_eq!(decode_utf8(&[0xED, 0xA0, 0x81]), None);
        // Invalid continuation byte.
        assert_eq!(decode_utf8(&[0xD4, 0xC2]), None);
        // Bad lengths
        assert_eq!(decode_utf8(&[0xC3]), None); // 2 bytes
        assert_eq!(decode_utf8(&[0xEF, 0xBF]), None); // 3 bytes
        assert_eq!(decode_utf8(&[0xF4, 0x8F, 0xBF]), None); // 4 bytes
                                                            // Not a minimal UTF-8 sequence
        assert_eq!(decode_utf8(&[TAG_TWO, TAG_CONT | b'a']), None);
        assert_eq!(decode_utf8(&[TAG_THREE, TAG_CONT, TAG_CONT | b'a']), None);
        assert_eq!(
            decode_utf8(&[TAG_FOUR, TAG_CONT, TAG_CONT, TAG_CONT | b'a',]),
            None
        );
    }

    #[test]
    fn reject_invalid_last() {
        // Invalid start byte
        assert_eq!(decode_last_utf8(&[0xFF]), None);
        // Surrogate pair
        assert_eq!(decode_last_utf8(&[0xED, 0xA0, 0x81]), None);
        // Bad lengths
        assert_eq!(decode_last_utf8(&[0xC3]), None); // 2 bytes
        assert_eq!(decode_last_utf8(&[0xEF, 0xBF]), None); // 3 bytes
        assert_eq!(decode_last_utf8(&[0xF4, 0x8F, 0xBF]), None); // 4 bytes
                                                                 // Not a minimal UTF-8 sequence
        assert_eq!(decode_last_utf8(&[TAG_TWO, TAG_CONT | b'a']), None);
        assert_eq!(
            decode_last_utf8(&[TAG_THREE, TAG_CONT, TAG_CONT | b'a',]),
            None
        );
        assert_eq!(
            decode_last_utf8(
                &[TAG_FOUR, TAG_CONT, TAG_CONT, TAG_CONT | b'a',]
            ),
            None
        );
    }
}
use std::{convert::TryInto, default, fmt::Debug, io::BufRead, iter::Peekable, str::Chars};

#[derive(Debug)]
#[repr(u8)]
enum TokenType {
    // Single-character tokens.
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,

    // oNE OR TWO CHARACTER TOKENS.
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    // lITERALS.
    Identifier(String),
    String(String),
    Number(f64),

    // kEYWORDS.
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    Eof,
    Unknown(char),
}
#[derive(Debug)]
struct Literal {}

#[derive(Debug)]
struct Token {
    t: TokenType,
    literal: Option<Literal>,
    line: u64,
}

struct Scanner<'a> {
    current_position: u32,
    it: Peekable<Chars<'a>>,
    end: bool,
    line: u64,
}

impl<'a> Scanner<'a> {
    fn new(source: &'a str) -> Self {
        Scanner {
            current_position: 0,
            it: source.chars().peekable(),
            end: false,
            line: 0,
        }
    }
}
impl Scanner<'_> {
    fn look_ahead(&mut self, c: char) -> bool {
        if self.it.peek().is_some() && self.it.peek().unwrap().to_owned() == c {
            self.it.next();
            true
        } else {
            false
        }
    }
    fn match_token(&mut self, c: char) -> Option<TokenType> {
        match c {
            '=' => Some(if self.look_ahead('=') {
                TokenType::EqualEqual
            } else {
                TokenType::Equal
            }),
            '!' => Some(if self.look_ahead('=') {
                TokenType::BangEqual
            } else {
                TokenType::Bang
            }),
            '<' => Some(if self.look_ahead('=') {
                TokenType::LessEqual
            } else {
                TokenType::Less
            }),
            '>' => Some(if self.look_ahead('=') {
                TokenType::GreaterEqual
            } else {
                TokenType::Greater
            }),
            ' ' => None,
            '/' => if self.look_ahead('/') {
                while self.it.peek() != None || self.it.peek() != Some(&'\n') {
                    self.it.next();
                }
                self.line+=1;
                None
            } else {
                Some(TokenType::Slash)
            },
            '"' => {
                let mut chars = vec![];
                
                while let Some(&n) = self.it.peek(){
                    if n == '"' {
                        break;
                    }
                    chars.push(n);
                }
                Some(TokenType::String(chars.into_iter().collect()))
            },
            '\n' => {self.line+=1; None},
            '\r' => None,
            '\t' => None,
            '.' => Some(TokenType::Dot),
            '(' => Some(TokenType::LeftParen),
            ')' => Some(TokenType::RightParen),
            '{' => Some(TokenType::LeftBrace),
            '}' => Some(TokenType::RightBrace),
            ',' => Some(TokenType::Comma),
            '-' => Some(TokenType::Minus),
            '+' => Some(TokenType::Plus),
            ';' => Some(TokenType::Semicolon),
            '*' => Some(TokenType::Star),
            _ => Some(TokenType::Unknown(c)),
        }
    }

    fn scan_token(&mut self, c: char) -> Option<Token> {
        if let Some(t) = self.match_token(c){
            Some(Token{t,literal:None,line:self.line})
        }else{
            None
        }
    }
}

impl Iterator for Scanner<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.it.next() {
            self.scan_token(c)
        } else if !self.end {
            self.end = true;
            Some(Token {
                t: TokenType::Eof,
                literal: None,
                line: self.line,
            })
        } else {
            None
        }
    }
}
use arrayfire::{homography, print, Array, Dim4};
use eframe::{
    egui::{
        self,
        plot::{Plot, PlotImage, Value},
    },
    epaint::ColorImage,
    epi,
};
use egui_extras::RetainedImage;
use image::{
    imageops::resize, io::Reader, DynamicImage, GenericImage, GenericImageView, ImageBuffer, Pixel,
    RgbaImage,
};
use imageproc::geometric_transformations::{warp_into, Projection};

pub struct MosaicApp {
    image_a: RetainedImage,
    image_b: RetainedImage,
    image_a_orig: DynamicImage,
    image_b_orig: DynamicImage,
    points_a: Vec<Value>,
    points_b: Vec<Value>,
    warped: Option<RetainedImage>,
    warped_orig: Option<DynamicImage>,
}

impl Default for MosaicApp {
    fn default() -> Self {
        let im1 = Reader::open("imgs/a.jpg").unwrap().decode().unwrap();
        let im2 = Reader::open("imgs/b.jpg").unwrap().decode().unwrap();
        Self {
            image_a: to_retained("image_a", im1.clone()),
            image_b: to_retained("image_b", im2.clone()),
            image_a_orig: im1,
            image_b_orig: im2,
            points_a: vec![],
            points_b: vec![],
            warped: None,
            warped_orig: None,
        }
    }
}

fn to_retained(debug_name: impl Into<String>, im: DynamicImage) -> RetainedImage {
    let size = [im.width() as _, im.height() as _];
    let mut pixels = im.to_rgba8();
    let pixels = pixels.as_flat_samples_mut();
    RetainedImage::from_color_image(
        debug_name,
        ColorImage::from_rgba_unmultiplied(size, pixels.as_slice()),
    )
}

fn clamp_add(a: u8, b: u8, max: u8) -> u8 {
    if (a as u16 + b as u16) > max.into() {
        max
    } else {
        a + b
    }
}

fn distance_alpha((a_x, a_y): (f64, f64), (b_x, b_y): (f64, f64), max: u32) -> u8 {
    255 - (((a_x - b_x).powf(2.0) + (a_y - b_y).powf(2.0)).sqrt() / max as f64) as u8
}

fn overlay_into(a: &DynamicImage, b: &mut DynamicImage, center: (f64, f64)) {
    let mut b_n = b.clone();
    println!("a-------");
    for y in 0..a.height() {
        for x in 0..a.width() {
            let mut p = a.get_pixel(x, y);
            let mut q = b.get_pixel(x, y);
            //p.0[3] = distance_alpha(center, (x as f64, y as f64), b.width());
            if p.0[3] == 0 {
                p = q;
            } else if q.0[3] != 0 {
                q.0[3] = 125;
                p.0[3] = 125;
                p.blend(&q);
            }
            b_n.put_pixel(x, y, p);
        }
    }
    b_n.save("dbg.jpg");
    let b_n_a: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n,
        b_n.width() / 8,
        b_n.height() / 8,
        image::imageops::FilterType::Nearest,
    ));
    println!("b-----");
    b_n_a.blur(100.0);
    let b_n_b: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n_a,
        b_n_a.width() / 8,
        b_n_a.height() / 8,
        image::imageops::FilterType::Nearest,
    ));
    b_n_b.blur(200.0);
    println!("b-----");
    let b_n_a: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n_a,
        b.width(),
        b.height(),
        image::imageops::FilterType::Nearest,
    ));
    let b_n_b: DynamicImage = image::DynamicImage::ImageRgba8(resize(
        &b_n_b,
        b.width(),
        b.height(),
        image::imageops::FilterType::Nearest,
    ));
    println!("c------");
    for y in 0..b.height() {
        for x in 0..b.width() {
            let mut p = b_n.get_pixel(x, y);
            let mut p_a = b_n_a.get_pixel(x, y);
            let mut p_b = b_n_b.get_pixel(x, y);
            let mut q = b.get_pixel(x, y);

            let mut r = if x < a.width() && y < a.height() {
                a.get_pixel(x, y)
            } else {
                image::Rgba([0, 0, 0, 0])
            };
            //if r.0[3] == 0 && q.0[3] != 0{
            //    p = q
            //}else if r.0[3] != 0 && q.0[3] == 0{
            //    p = r;
            //}else{
            p_a.0[3] = 185;
            // Smallest
            p_b.0[3] = 125;

            // Blend all three photos together
            p_a.blend(&p_b);
None            p.blend(&p_a);
            // Set alpha according to distance from center
            p.0[3] = distance_alpha(center, (x as f64, y as f64), b.width());

            // Blend first photo and all merged photos
            if r.0[3] != 0 {
                r.0[3] = 150;
                p.blend(&r);
            }
            p.0[3] = 255;

            
            //}

            b.put_pixel(x, y, p);
        }
    }
    println!("d------");
}

fn find_homography(a: Vec<Value>, b: Vec<Value>) -> [f32; 9] {
    let mut v = [1.0; 9];
    let mut x_src = [0.0; 4];
    let mut y_src = [0.0; 4];
    let mut x_dst = [0.0; 4];
    let mut y_dst = [0.0; 4];
    for i in 0..a.len() {
        x_src[i] = a[i].x as f32;
        y_src[i] = a[i].y as f32;
        x_dst[i] = b[i].x as f32;
        y_dst[i] = b[i].y as f32;
    }
    let x_src = Array::new(&x_src, Dim4::new(&[4, 1, 1, 1]));
    let y_src = Array::new(&y_src, Dim4::new(&[4, 1, 1, 1]));
    let x_dst = Array::new(&x_dst, Dim4::new(&[4, 1, 1, 1]));
    let y_dst = Array::new(&y_dst, Dim4::new(&[4, 1, 1, 1]));
    let (h, i): (Array<f32>, i32) = homography(
        &x_src,
        &y_src,
        &x_dst,
        &y_dst,
        arrayfire::HomographyType::RANSAC,
        100000.0,
        10,
    );

    print(&h);
    h.host(&mut v);
    v
}

impl epi::App for MosaicApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let plot_image_a = PlotImage::new(
                self.image_a.texture_id(ctx),
                egui::plot::Value {
                    x: (self.image_a.size_vec2().x / 2.0) as f64,
                    y: (self.image_a.size_vec2().y / 2.0) as f64,
                },
                self.image_a.size_vec2(),
            );

            let plot_image_b = PlotImage::new(
                self.image_b.texture_id(ctx),
                egui::plot::Value {
                    x: (self.image_b.size_vec2().x / 2.0) as f64,
                    y: (self.image_b.size_vec2().y / 2.0) as f64,
                },
                self.image_b.size_vec2(),
            );
            let plot_a = Plot::new("image_a_plot");
            let plot_b = Plot::new("image_b_plot");
            let plot_c = Plot::new("image_c_plot");
            //let img_plot = PlotImage::new(texture_id, center_position, size)

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    plot_a
                        .allow_drag(false)
                        .show_axes([false, false])
                        .height(800.0)
                        .width(800.0)
                        .show(ui, |plot_ui| {
                            plot_ui.image(plot_image_a.name("image_a"));
                            if plot_ui.plot_clicked() {
                                let mut coord = plot_ui.pointer_coordinate().unwrap();
                                coord.y = self.image_a_orig.height() as f64 - coord.y;
                                self.points_a.insert(0, coord);
                                if self.points_a.len() > 4 {
                                    self.points_a.pop();
                                }
                            }
                        });
                    plot_b
                        .allow_drag(false)
                        .show_axes([false, false])
                        .height(800.0)
                        .width(800.0)
                        .show(ui, |plot_ui| {
                            plot_ui.image(plot_image_b.name("image_b"));
                            if plot_ui.plot_clicked() {
                                let mut coord = plot_ui.pointer_coordinate().unwrap();
                                coord.y = self.image_b_orig.height() as f64 - coord.y;
                                self.points_b.insert(0, coord);
                                if self.points_b.len() > 4 {
                                    self.points_b.pop();
                                }
                            }
                        });
                });
                if self.warped.is_some() {
                    if ui.button("save").clicked() {
                       self.warped_orig.clone().unwrap().save("out.jpg");

                    }
                    let plot_image_c = PlotImage::new(
                        self.warped.as_ref().unwrap().texture_id(ctx),
                        egui::plot::Value {
                            x: (self.image_b.size_vec2().x / 2.0) as f64,
                            y: (self.image_b.size_vec2().y / 2.0) as f64,
                        },
                        self.image_b.size_vec2(),
                    );
                    plot_c
                        .allow_drag(false)
                        .show_axes([false, false])
                        .height(800.0)
                        .width(1600.0)
                        .show(ui, |plot_ui| {
                            plot_ui.image(plot_image_c.name("image_c"));
                        });
                }
            });
            if ui.button("Merge").clicked() {
                if self.points_a.len() == 4 && self.points_b.len() == 4 {
                    let h = find_homography(self.points_b.clone(), self.points_a.clone());
                    let projection = Projection::from_matrix(h).unwrap();
                    let white: image::Rgba<u8> = image::Rgba([0, 0, 0, 0]);
                    let mut canvas: RgbaImage =
                        ImageBuffer::new(self.image_a_orig.width() * 2, self.image_a_orig.height());
                    warp_into(
                        &self.image_b_orig.to_rgba8(),
                        &projection,
                        imageproc::geometric_transformations::Interpolation::Nearest,
                        white,
                        &mut canvas,
                    );
                    let mut canvas = image::DynamicImage::ImageRgba8(canvas);
                    let x = self
                        .points_a
                        .clone()
                        .iter()
                        .fold(0.0, |cntr, curr| cntr + curr.x)
                        / 4.0;
                    let y = self
                        .points_a
                        .clone()
                        .iter()
                        .fold(0.0, |cntr, curr| cntr + curr.y)
                        / 4.0;
                    overlay_into(&self.image_a_orig, &mut canvas, (x, y));
                    self.warped_orig = Some(canvas.clone());
                    self.warped = Some(to_retained("w", canvas));
                    self.points_a = vec![];
                    self.points_b = vec![];
                }
            }
            egui::warn_if_debug_build(ui);
        });
    }
}
#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::MosaicApp;

// ----------------------------------------------------------------------------
// When compiling for web:

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), eframe::wasm_bindgen::JsValue> {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let app = MosaicApp::default();
    eframe::start_web(canvas_id, Box::new(app))
}
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use once_cell::unsync::OnceCell;

use syntect::highlighting::Theme;
use syntect::parsing::{SyntaxReference, SyntaxSet};

use path_abs::PathAbs;

use crate::error::*;
use crate::input::{InputReader, OpenedInput};
use crate::syntax_mapping::ignored_suffixes::IgnoredSuffixes;
use crate::syntax_mapping::MappingTarget;
use crate::{bat_warning, SyntaxMapping};

use lazy_theme_set::LazyThemeSet;

use serialized_syntax_set::*;

#[cfg(feature = "build-assets")]
pub use crate::assets::build_assets::*;

pub(crate) mod assets_metadata;
#[cfg(feature = "build-assets")]
mod build_assets;
mod lazy_theme_set;
mod serialized_syntax_set;

#[derive(Debug)]
pub struct HighlightingAssets {
    syntax_set_cell: OnceCell<SyntaxSet>,
    serialized_syntax_set: SerializedSyntaxSet,

    theme_set: LazyThemeSet,
    fallback_theme: Option<&'static str>,
}

#[derive(Debug)]
pub struct SyntaxReferenceInSet<'a> {
    pub syntax: &'a SyntaxReference,
    pub syntax_set: &'a SyntaxSet,
}

/// Compress for size of ~700 kB instead of ~4600 kB at the cost of ~30% longer deserialization time
pub(crate) const COMPRESS_SYNTAXES: bool = true;

/// We don't want to compress our [LazyThemeSet] since the lazy-loaded themes
/// within it are already compressed, and compressing another time just makes
/// performance suffer
pub(crate) const COMPRESS_THEMES: bool = false;

/// Compress for size of ~40 kB instead of ~200 kB without much difference in
/// performance due to lazy-loading
pub(crate) const COMPRESS_LAZY_THEMES: bool = true;

/// Compress for size of ~10 kB instead of ~120 kB
pub(crate) const COMPRESS_ACKNOWLEDGEMENTS: bool = true;

impl HighlightingAssets {
    fn new(serialized_syntax_set: SerializedSyntaxSet, theme_set: LazyThemeSet) -> Self {
        HighlightingAssets {
            syntax_set_cell: OnceCell::new(),
            serialized_syntax_set,
            theme_set,
            fallback_theme: None,
        }
    }

    pub fn default_theme() -> &'static str {
        "Monokai Extended"
    }

    pub fn from_cache(cache_path: &Path) -> Result<Self> {
        Ok(HighlightingAssets::new(
            SerializedSyntaxSet::FromFile(cache_path.join("syntaxes.bin")),
            asset_from_cache(&cache_path.join("themes.bin"), "theme set", COMPRESS_THEMES)?,
        ))
    }

    pub fn from_binary() -> Self {
        HighlightingAssets::new(
            SerializedSyntaxSet::FromBinary(get_serialized_integrated_syntaxset()),
            get_integrated_themeset(),
        )
    }

    pub fn set_fallback_theme(&mut self, theme: &'static str) {
        self.fallback_theme = Some(theme);
    }

    /// Return the collection of syntect syntax definitions.
    pub fn get_syntax_set(&self) -> Result<&SyntaxSet> {
        self.syntax_set_cell
            .get_or_try_init(|| self.serialized_syntax_set.deserialize())
    }

    /// Use [Self::get_syntaxes] instead
    #[deprecated]
    pub fn syntaxes(&self) -> &[SyntaxReference] {
        self.get_syntax_set()
            .expect(".syntaxes() is deprecated, use .get_syntaxes() instead")
            .syntaxes()
    }

    pub fn get_syntaxes(&self) -> Result<&[SyntaxReference]> {
        Ok(self.get_syntax_set()?.syntaxes())
    }

    fn get_theme_set(&self) -> &LazyThemeSet {
        &self.theme_set
    }

    pub fn themes(&self) -> impl Iterator<Item = &str> {
        self.get_theme_set().themes()
    }

    /// Use [Self::get_syntax_for_path] instead
    #[deprecated]
    pub fn syntax_for_file_name(
        &self,
        file_name: impl AsRef<Path>,
        mapping: &SyntaxMapping,
    ) -> Option<&SyntaxReference> {
        self.get_syntax_for_path(file_name, mapping)
            .ok()
            .map(|syntax_in_set| syntax_in_set.syntax)
    }

    /// Detect the syntax based on, in order:
    ///  1. Syntax mappings with [MappingTarget::MapTo] and [MappingTarget::MapToUnknown]
    ///     (e.g. `/etc/profile` -> `Bourne Again Shell (bash)`)
    ///  2. The file name (e.g. `Dockerfile`)
    ///  3. Syntax mappings with [MappingTarget::MapExtensionToUnknown]
    ///     (e.g. `*.conf`)
    ///  4. The file name extension (e.g. `.rs`)
    ///
    /// When detecting syntax based on syntax mappings, the full path is taken
    /// into account. When detecting syntax based on file name, no regard is
    /// taken to the path of the file. Only the file name itself matters. When
    /// detecting syntax based on file name extension, only the file name
    /// extension itself matters.
    ///
    /// Returns [Error::UndetectedSyntax] if it was not possible detect syntax
    /// based on path/file name/extension (or if the path was mapped to
    /// [MappingTarget::MapToUnknown] or [MappingTarget::MapExtensionToUnknown]).
    /// In this case it is appropriate to fall back to other methods to detect
    /// syntax. Such as using the contents of the first line of the file.
    ///
    /// Returns [Error::UnknownSyntax] if a syntax mapping exist, but the mapped
    /// syntax does not exist.
    pub fn get_syntax_for_path(
        &self,
        path: impl AsRef<Path>,
        mapping: &SyntaxMapping,
    ) -> Result<SyntaxReferenceInSet> {
        let path = path.as_ref();

        let syntax_match = mapping.get_syntax_for(path);

        if let Some(MappingTarget::MapToUnknown) = syntax_match {
            return Err(Error::UndetectedSyntax(path.to_string_lossy().into()));
        }

        if let Some(MappingTarget::MapTo(syntax_name)) = syntax_match {
            return self
                .find_syntax_by_name(syntax_name)?
                .ok_or_else(|| Error::UnknownSyntax(syntax_name.to_owned()));
        }

        let file_name = path.file_name().unwrap_or_default();

        match (
            self.get_syntax_for_file_name(file_name, &mapping.ignored_suffixes)?,
            syntax_match,
        ) {
            (Some(syntax), _) => Ok(syntax),

            (_, Some(MappingTarget::MapExtensionToUnknown)) => {
                Err(Error::UndetectedSyntax(path.to_string_lossy().into()))
            }

            _ => self
                .get_syntax_for_file_extension(file_name, &mapping.ignored_suffixes)?
                .ok_or_else(|| Error::UndetectedSyntax(path.to_string_lossy().into())),
        }
    }

    /// Look up a syntect theme by name.
    pub fn get_theme(&self, theme: &str) -> &Theme {
        match self.get_theme_set().get(theme) {
            Some(theme) => theme,
            None => {
                if theme == "ansi-light" || theme == "ansi-dark" {
                    bat_warning!("Theme '{}' is deprecated, using 'ansi' instead.", theme);
                    return self.get_theme("ansi");
                }
                if !theme.is_empty() {
                    bat_warning!("Unknown theme '{}', using default.", theme)
                }
                self.get_theme_set()
                    .get(self.fallback_theme.unwrap_or_else(Self::default_theme))
                    .expect("something is very wrong if the default theme is missing")
            }
        }
    }

    pub(crate) fn get_syntax(
        &self,
        language: Option<&str>,
        input: &mut OpenedInput,
        mapping: &SyntaxMapping,
    ) -> Result<SyntaxReferenceInSet> {
        if let Some(language) = language {
            let syntax_set = self.get_syntax_set()?;
            return syntax_set
                .find_syntax_by_token(language)
                .map(|syntax| SyntaxReferenceInSet { syntax, syntax_set })
                .ok_or_else(|| Error::UnknownSyntax(language.to_owned()));
        }

        let path = input.path();
        let path_syntax = if let Some(path) = path {
            self.get_syntax_for_path(
                PathAbs::new(path).map_or_else(|_| path.to_owned(), |p| p.as_path().to_path_buf()),
                mapping,
            )
        } else {
            Err(Error::UndetectedSyntax("[unknown]".into()))
        };

        match path_syntax {
            // If a path wasn't provided, or if path based syntax detection
            // above failed, we fall back to first-line syntax detection.
            Err(Error::UndetectedSyntax(path)) => self
                .get_first_line_syntax(&mut input.reader)?
                .ok_or(Error::UndetectedSyntax(path)),
            _ => path_syntax,
        }
    }

    pub(crate) fn find_syntax_by_name(
        &self,
        syntax_name: &str,
    ) -> Result<Option<SyntaxReferenceInSet>> {
        let syntax_set = self.get_syntax_set()?;
        Ok(syntax_set
            .find_syntax_by_name(syntax_name)
            .map(|syntax| SyntaxReferenceInSet { syntax, syntax_set }))
    }

    fn find_syntax_by_extension(&self, e: Option<&OsStr>) -> Result<Option<SyntaxReferenceInSet>> {
        let syntax_set = self.get_syntax_set()?;
        let extension = e.and_then(|x| x.to_str()).unwrap_or_default();
        Ok(syntax_set
            .find_syntax_by_extension(extension)
            .map(|syntax| SyntaxReferenceInSet { syntax, syntax_set }))
    }

    fn get_syntax_for_file_name(
        &self,
        file_name: &OsStr,
        ignored_suffixes: &IgnoredSuffixes,
    ) -> Result<Option<SyntaxReferenceInSet>> {
        let mut syntax = self.find_syntax_by_extension(Some(file_name))?;
        if syntax.is_none() {
            syntax =
                ignored_suffixes.try_with_stripped_suffix(file_name, |stripped_file_name| {
                    // Note: recursion
                    self.get_syntax_for_file_name(stripped_file_name, ignored_suffixes)
                })?;
        }
        Ok(syntax)
    }

    fn get_syntax_for_file_extension(
        &self,
        file_name: &OsStr,
        ignored_suffixes: &IgnoredSuffixes,
    ) -> Result<Option<SyntaxReferenceInSet>> {
        let mut syntax = self.find_syntax_by_extension(Path::new(file_name).extension())?;
        if syntax.is_none() {
            syntax =
                ignored_suffixes.try_with_stripped_suffix(file_name, |stripped_file_name| {
                    // Note: recursion
                    self.get_syntax_for_file_extension(stripped_file_name, ignored_suffixes)
                })?;
        }
        Ok(syntax)
    }

    fn get_first_line_syntax(
        &self,
        reader: &mut InputReader,
    ) -> Result<Option<SyntaxReferenceInSet>> {
        let syntax_set = self.get_syntax_set()?;
        Ok(String::from_utf8(reader.first_line.clone())
            .ok()
            .and_then(|l| syntax_set.find_syntax_by_first_line(&l))
            .map(|syntax| SyntaxReferenceInSet { syntax, syntax_set }))
    }
}

pub(crate) fn get_serialized_integrated_syntaxset() -> &'static [u8] {
    include_bytes!("../assets/syntaxes.bin")
}

pub(crate) fn get_integrated_themeset() -> LazyThemeSet {
    from_binary(include_bytes!("../assets/themes.bin"), COMPRESS_THEMES)
}

pub fn get_acknowledgements() -> String {
    from_binary(
        include_bytes!("../assets/acknowledgements.bin"),
        COMPRESS_ACKNOWLEDGEMENTS,
    )
}

pub(crate) fn from_binary<T: serde::de::DeserializeOwned>(v: &[u8], compressed: bool) -> T {
    asset_from_contents(v, "n/a", compressed)
        .expect("data integrated in binary is never faulty, but make sure `compressed` is in sync!")
}

fn asset_from_contents<T: serde::de::DeserializeOwned>(
    contents: &[u8],
    description: &str,
    compressed: bool,
) -> Result<T> {
    if compressed {
        bincode::deserialize_from(flate2::read::ZlibDecoder::new(contents))
    } else {
        bincode::deserialize_from(contents)
    }
    .map_err(|_| format!("Could not parse {}", description).into())
}

fn asset_from_cache<T: serde::de::DeserializeOwned>(
    path: &Path,
    description: &str,
    compressed: bool,
) -> Result<T> {
    let contents = fs::read(path).map_err(|_| {
        format!(
            "Could not load cached {} '{}'",
            description,
            path.to_string_lossy()
        )
    })?;
    asset_from_contents(&contents[..], description, compressed)
        .map_err(|_| format!("Could not parse cached {}", description).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::OsStr;

    use std::fs::File;
    use std::io::{BufReader, Write};
    use tempfile::TempDir;

    use crate::input::Input;

    struct SyntaxDetectionTest<'a> {
        assets: HighlightingAssets,
        pub syntax_mapping: SyntaxMapping<'a>,
        pub temp_dir: TempDir,
    }

    impl<'a> SyntaxDetectionTest<'a> {
        fn new() -> Self {
            SyntaxDetectionTest {
                assets: HighlightingAssets::from_binary(),
                syntax_mapping: SyntaxMapping::builtin(),
                temp_dir: TempDir::new().expect("creation of temporary directory"),
            }
        }

        fn get_syntax_name(
            &self,
            language: Option<&str>,
            input: &mut OpenedInput,
            mapping: &SyntaxMapping,
        ) -> String {
            self.assets
                .get_syntax(language, input, mapping)
                .map(|syntax_in_set| syntax_in_set.syntax.name.clone())
                .unwrap_or_else(|_| "!no syntax!".to_owned())
        }

        fn syntax_for_real_file_with_content_os(
            &self,
            file_name: &OsStr,
            first_line: &str,
        ) -> String {
            let file_path = self.temp_dir.path().join(file_name);
            {
                let mut temp_file = File::create(&file_path).unwrap();
                writeln!(temp_file, "{}", first_line).unwrap();
            }

            let input = Input::ordinary_file(&file_path);
            let dummy_stdin: &[u8] = &[];
            let mut opened_input = input.open(dummy_stdin, None).unwrap();

            self.get_syntax_name(None, &mut opened_input, &self.syntax_mapping)
        }

        fn syntax_for_file_with_content_os(&self, file_name: &OsStr, first_line: &str) -> String {
            let file_path = self.temp_dir.path().join(file_name);
            let input = Input::from_reader(Box::new(BufReader::new(first_line.as_bytes())))
                .with_name(Some(&file_path));
            let dummy_stdin: &[u8] = &[];
            let mut opened_input = input.open(dummy_stdin, None).unwrap();

            self.get_syntax_name(None, &mut opened_input, &self.syntax_mapping)
        }

        #[cfg(unix)]
        fn syntax_for_file_os(&self, file_name: &OsStr) -> String {
            self.syntax_for_file_with_content_os(file_name, "")
        }

        fn syntax_for_file_with_content(&self, file_name: &str, first_line: &str) -> String {
            self.syntax_for_file_with_content_os(OsStr::new(file_name), first_line)
        }

        fn syntax_for_file(&self, file_name: &str) -> String {
            self.syntax_for_file_with_content(file_name, "")
        }

        fn syntax_for_stdin_with_content(&self, file_name: &str, content: &[u8]) -> String {
            let input = Input::stdin().with_name(Some(file_name));
            let mut opened_input = input.open(content, None).unwrap();

            self.get_syntax_name(None, &mut opened_input, &self.syntax_mapping)
        }

        fn syntax_is_same_for_inputkinds(&self, file_name: &str, content: &str) -> bool {
            let as_file = self.syntax_for_real_file_with_content_os(file_name.as_ref(), content);
            let as_reader = self.syntax_for_file_with_content_os(file_name.as_ref(), content);
            let consistent = as_file == as_reader;
            // TODO: Compare StdIn somehow?

            if !consistent {
                eprintln!(
                    "Inconsistent syntax detection:\nFor File: {}\nFor Reader: {}",
                    as_file, as_reader
                )
            }

            consistent
        }
    }

    #[test]
    fn syntax_detection_basic() {
        let test = SyntaxDetectionTest::new();

        assert_eq!(test.syntax_for_file("test.rs"), "Rust");
        assert_eq!(test.syntax_for_file("test.cpp"), "C++");
        assert_eq!(test.syntax_for_file("test.build"), "NAnt Build File");
        assert_eq!(
            test.syntax_for_file("PKGBUILD"),
            "Bourne Again Shell (bash)"
        );
        assert_eq!(test.syntax_for_file(".bashrc"), "Bourne Again Shell (bash)");
        assert_eq!(test.syntax_for_file("Makefile"), "Makefile");
    }

    #[cfg(unix)]
    #[test]
    fn syntax_detection_invalid_utf8() {
        use std::os::unix::ffi::OsStrExt;

        let test = SyntaxDetectionTest::new();

        assert_eq!(
            test.syntax_for_file_os(OsStr::from_bytes(b"invalid_\xFEutf8_filename.rs")),
            "Rust"
        );
    }

    #[test]
    fn syntax_detection_same_for_inputkinds() {
        let mut test = SyntaxDetectionTest::new();

        test.syntax_mapping
            .insert("*.myext", MappingTarget::MapTo("C"))
            .ok();
        test.syntax_mapping
            .insert("MY_FILE", MappingTarget::MapTo("Markdown"))
            .ok();

        assert!(test.syntax_is_same_for_inputkinds("Test.md", ""));
        assert!(test.syntax_is_same_for_inputkinds("Test.txt", "#!/bin/bash"));
        assert!(test.syntax_is_same_for_inputkinds(".bashrc", ""));
        assert!(test.syntax_is_same_for_inputkinds("test.h", ""));
        assert!(test.syntax_is_same_for_inputkinds("test.js", "#!/bin/bash"));
        assert!(test.syntax_is_same_for_inputkinds("test.myext", ""));
        assert!(test.syntax_is_same_for_inputkinds("MY_FILE", ""));
        assert!(test.syntax_is_same_for_inputkinds("MY_FILE", "<?php"));
    }

    #[test]
    fn syntax_detection_well_defined_mapping_for_duplicate_extensions() {
        let test = SyntaxDetectionTest::new();

        assert_eq!(test.syntax_for_file("test.h"), "C++");
        assert_eq!(test.syntax_for_file("test.sass"), "Sass");
        assert_eq!(test.syntax_for_file("test.js"), "JavaScript (Babel)");
        assert_eq!(test.syntax_for_file("test.fs"), "F#");
        assert_eq!(test.syntax_for_file("test.v"), "Verilog");
    }

    #[test]
    fn syntax_detection_first_line() {
        let test = SyntaxDetectionTest::new();

        assert_eq!(
            test.syntax_for_file_with_content("my_script", "#!/bin/bash"),
            "Bourne Again Shell (bash)"
        );
        assert_eq!(
            test.syntax_for_file_with_content("build", "#!/bin/bash"),
            "Bourne Again Shell (bash)"
        );
        assert_eq!(
            test.syntax_for_file_with_content("my_script", "<?php"),
            "PHP"
        );
    }

    #[test]
    fn syntax_detection_with_custom_mapping() {
        let mut test = SyntaxDetectionTest::new();

        assert_eq!(test.syntax_for_file("test.h"), "C++");
        test.syntax_mapping
            .insert("*.h", MappingTarget::MapTo("C"))
            .ok();
        assert_eq!(test.syntax_for_file("test.h"), "C");
    }

    #[test]
    fn syntax_detection_with_extension_mapping_to_unknown() {
        let mut test = SyntaxDetectionTest::new();

        // Normally, a CMakeLists.txt file shall use the CMake syntax, even if it is
        // a bash script in disguise
        assert_eq!(
            test.syntax_for_file_with_content("CMakeLists.txt", "#!/bin/bash"),
            "CMake"
        );

        // Other .txt files shall use the Plain Text syntax
        assert_eq!(
            test.syntax_for_file_with_content("some-other.txt", "#!/bin/bash"),
            "Plain Text"
        );

        // If we setup MapExtensionToUnknown on *.txt, the match on the full
        // file name of "CMakeLists.txt" shall have higher prio, and CMake shall
        // still be used for it
        test.syntax_mapping
            .insert("*.txt", MappingTarget::MapExtensionToUnknown)
            .ok();
        assert_eq!(
            test.syntax_for_file_with_content("CMakeLists.txt", "#!/bin/bash"),
            "CMake"
        );

        // However, for *other* files with a .txt extension, first-line fallback
        // shall now be used
        assert_eq!(
            test.syntax_for_file_with_content("some-other.txt", "#!/bin/bash"),
            "Bourne Again Shell (bash)"
        );
    }

    #[test]
    fn syntax_detection_is_case_sensitive() {
        let mut test = SyntaxDetectionTest::new();

        assert_ne!(test.syntax_for_file("README.MD"), "Markdown");
        test.syntax_mapping
            .insert("*.MD", MappingTarget::MapTo("Markdown"))
            .ok();
        assert_eq!(test.syntax_for_file("README.MD"), "Markdown");
    }

    #[test]
    fn syntax_detection_stdin_filename() {
        let test = SyntaxDetectionTest::new();

        // from file extension
        assert_eq!(test.syntax_for_stdin_with_content("test.cpp", b"a"), "C++");
        // from first line (fallback)
        assert_eq!(
            test.syntax_for_stdin_with_content("my_script", b"#!/bin/bash"),
            "Bourne Again Shell (bash)"
        );
    }

    #[cfg(unix)]
    #[test]
    fn syntax_detection_for_symlinked_file() {
        use std::os::unix::fs::symlink;

        let test = SyntaxDetectionTest::new();
        let file_path = test.temp_dir.path().join("my_ssh_config_filename");
        {
            File::create(&file_path).unwrap();
        }
        let file_path_symlink = test.temp_dir.path().join(".ssh").join("config");

        std::fs::create_dir(test.temp_dir.path().join(".ssh"))
            .expect("creation of directory succeeds");
        symlink(&file_path, &file_path_symlink).expect("creation of symbolic link succeeds");

        let input = Input::ordinary_file(&file_path_symlink);
        let dummy_stdin: &[u8] = &[];
        let mut opened_input = input.open(dummy_stdin, None).unwrap();

        assert_eq!(
            test.get_syntax_name(None, &mut opened_input, &test.syntax_mapping),
            "SSH Config"
        );
    }
}
use crate::line_range::{HighlightedLineRanges, LineRanges};
#[cfg(feature = "paging")]
use crate::paging::PagingMode;
use crate::style::StyleComponents;
use crate::syntax_mapping::SyntaxMapping;
use crate::wrapping::WrappingMode;

#[derive(Debug, Clone)]
pub enum VisibleLines {
    /// Show all lines which are included in the line ranges
    Ranges(LineRanges),

    #[cfg(feature = "git")]
    /// Only show lines surrounding added/deleted/modified lines
    DiffContext(usize),
}

impl VisibleLines {
    pub fn diff_mode(&self) -> bool {
        match self {
            Self::Ranges(_) => false,
            #[cfg(feature = "git")]
            Self::DiffContext(_) => true,
        }
    }
}

impl Default for VisibleLines {
    fn default() -> Self {
        VisibleLines::Ranges(LineRanges::default())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Config<'a> {
    /// The explicitly configured language, if any
    pub language: Option<&'a str>,

    /// Whether or not to show/replace non-printable characters like space, tab and newline.
    pub show_nonprintable: bool,

    /// The character width of the terminal
    pub term_width: usize,

    /// The width of tab characters.
    /// Currently, a value of 0 will cause tabs to be passed through without expanding them.
    pub tab_width: usize,

    /// Whether or not to simply loop through all input (`cat` mode)
    pub loop_through: bool,

    /// Whether or not the output should be colorized
    pub colored_output: bool,

    /// Whether or not the output terminal supports true color
    pub true_color: bool,

    /// Style elements (grid, line numbers, ...)
    pub style_components: StyleComponents,

    /// If and how text should be wrapped
    pub wrapping_mode: WrappingMode,

    /// Pager or STDOUT
    #[cfg(feature = "paging")]
    pub paging_mode: PagingMode,

    /// Specifies which lines should be printed
    pub visible_lines: VisibleLines,

    /// The syntax highlighting theme
    pub theme: String,

    /// File extension/name mappings
    pub syntax_mapping: SyntaxMapping<'a>,

    /// Command to start the pager
    pub pager: Option<&'a str>,

    /// Whether or not to use ANSI italics
    pub use_italic_text: bool,

    /// Ranges of lines which should be highlighted with a special background color
    pub highlighted_lines: HighlightedLineRanges,

    /// Whether or not to allow custom assets. If this is false or if custom assets (a.k.a.
    /// cached assets) are not available, assets from the binary will be used instead.
    pub use_custom_assets: bool,
}

#[cfg(all(feature = "minimal-application", feature = "paging"))]
pub fn get_pager_executable(config_pager: Option<&str>) -> Option<String> {
    crate::pager::get_pager(config_pager)
        .ok()
        .flatten()
        .map(|pager| pager.bin)
}

#[test]
fn default_config_should_include_all_lines() {
    use crate::line_range::RangeCheckResult;

    assert_eq!(LineRanges::default().check(17), RangeCheckResult::InRange);
}

#[test]
fn default_config_should_highlight_no_lines() {
    use crate::line_range::RangeCheckResult;

    assert_ne!(
        Config::default().highlighted_lines.0.check(17),
        RangeCheckResult::InRange
    );
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PagingMode {
    Always,
    QuitIfOneScreen,
    Never,
}

impl Default for PagingMode {
    fn default() -> Self {
        PagingMode::Never
    }
}
pub trait Model{
    fn test(&self){
        println!("Model trait!");
    }
}
use s3::bucket::Bucket;
use s3::creds::Credentials;
use s3::region::Region;
use std::path::Path;

use freefs::{
    datasource::sources::s3_bucket::BucketSource, files::Files, Storage,
};
use keys_grpc_rs::KeysManager;

fn main() {
    let bb = Storage {
        name: "backblaze".to_string(),
        region: Region::Custom {
            region: "us-west-002".to_string(),
            endpoint: "https://s3.us-west-002.backblazeb2.com".to_string(),
        },
        credentials: Credentials::default_blocking().unwrap(),
        bucket: "rust-test".to_string(),
    };
    let mut manager = KeysManager::new("freefs".to_string());
    //manager.auth_setup();
    manager.auth_unlock();
    let bucket = create_bucket(bb);
    let files = bucket.list_blocking("test-folder".to_string(), None).unwrap();
    for (i, (file,code)) in files.iter().enumerate(){
        assert_eq!(&200, code);
        let contents = &file.contents;
        for obj in contents{
            println!("{:?}",obj.key);
        }
    }
    let mountpoint = Path::new("/tmp/rust-test");
    let data_source = BucketSource::new(bucket,"/tmp/rust-test-transient".to_string(),"/tmp/rust-test-stage".to_string(),manager);
    let fs = Files::new(data_source);
    mount(mountpoint,fs);
}
fn mount(mountpoint: &Path, fs: Files) {
    let result = fuse::mount(fs, &mountpoint, &[]).unwrap();
    println!("{:?}", result);
}

fn create_bucket(storage: Storage) -> Bucket {
    Bucket::new(&storage.bucket, storage.region, storage.credentials).unwrap()
}
// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

/*!

[grpcio] is a Rust implementation of [gRPC], which is a high performance, open source universal RPC
framework that puts mobile and HTTP/2 first. grpcio is built on [gRPC Core] and [futures-rs].

[grpcio]: https://github.com/tikv/grpc-rs/
[gRPC]: https://grpc.io/
[gRPC Core]: https://github.com/grpc/grpc
[futures-rs]: https://github.com/rust-lang/futures-rs

## Optional features

- **`secure`** *(enabled by default)* - Enables support for TLS encryption and some authentication
  mechanisms.

*/

#![allow(clippy::new_without_default)]
#![allow(clippy::new_without_default)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::option_map_unit_fn)]

use grpcio_sys as grpc_sys;
#[macro_use]
extern crate log;

mod auth_context;
mod buf;
mod call;
mod channel;
mod client;
mod codec;
mod cq;
mod env;
mod error;
mod log_util;
mod metadata;
mod quota;
#[cfg(feature = "secure")]
mod security;
mod server;
mod task;

pub use crate::call::client::{
    CallOption, ClientCStreamReceiver, ClientCStreamSender, ClientDuplexReceiver,
    ClientDuplexSender, ClientSStreamReceiver, ClientUnaryReceiver, StreamingCallSink,
};
pub use crate::call::server::{
    ClientStreamingSink, ClientStreamingSinkResult, Deadline, DuplexSink, DuplexSinkFailure,
    RequestStream, RpcContext, ServerStreamingSink, ServerStreamingSinkFailure, UnarySink,
    UnarySinkResult,
};
pub use crate::call::{MessageReader, Method, MethodType, RpcStatus, RpcStatusCode, WriteFlags};
pub use crate::channel::{
    Channel, ChannelBuilder, CompressionAlgorithms, CompressionLevel, ConnectivityState, LbPolicy,
    OptTarget,
};
pub use crate::client::Client;

#[cfg(feature = "protobuf-codec")]
pub use crate::codec::pb_codec::{de as pb_de, ser as pb_ser};
#[cfg(feature = "prost-codec")]
pub use crate::codec::pr_codec::{de as pr_de, ser as pr_ser};

pub use crate::auth_context::{AuthContext, AuthProperty, AuthPropertyIter};
pub use crate::codec::Marshaller;
pub use crate::env::{EnvBuilder, Environment};
pub use crate::error::{Error, Result};
pub use crate::log_util::redirect_log;
pub use crate::metadata::{Metadata, MetadataBuilder, MetadataIter};
pub use crate::quota::ResourceQuota;
#[cfg(feature = "secure")]
pub use crate::security::{
    CertificateRequestType, ChannelCredentials, ChannelCredentialsBuilder, ServerCredentials,
    ServerCredentialsBuilder, ServerCredentialsFetcher,
};
pub use crate::server::{Server, ServerBuilder, Service, ServiceBuilder, ShutdownFuture};
// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{Builder as ThreadBuilder, JoinHandle};

use crate::grpc_sys;

use crate::cq::{CompletionQueue, CompletionQueueHandle, EventType, WorkQueue};
use crate::task::CallTag;

// event loop
fn poll_queue(tx: mpsc::Sender<CompletionQueue>) {
    let cq = Arc::new(CompletionQueueHandle::new());
    let worker_info = Arc::new(WorkQueue::new());
    let cq = CompletionQueue::new(cq, worker_info);
    tx.send(cq.clone()).expect("send back completion queue");
    loop {
        let e = cq.next();
        match e.type_ {
            EventType::GRPC_QUEUE_SHUTDOWN => break,
            // timeout should not happen in theory.
            EventType::GRPC_QUEUE_TIMEOUT => continue,
            EventType::GRPC_OP_COMPLETE => {}
        }

        let tag: Box<CallTag> = unsafe { Box::from_raw(e.tag as _) };

        tag.resolve(&cq, e.success != 0);
        while let Some(work) = unsafe { cq.worker.pop_work() } {
            work.finish();
        }
    }
}

/// [`Environment`] factory in order to configure the properties.
pub struct EnvBuilder {
    cq_count: usize,
    name_prefix: Option<String>,
}

impl EnvBuilder {
    /// Initialize a new [`EnvBuilder`].
    pub fn new() -> EnvBuilder {
        EnvBuilder {
            cq_count: unsafe { grpc_sys::gpr_cpu_num_cores() as usize },
            name_prefix: None,
        }
    }

    /// Set the number of completion queues and polling threads. Each thread polls
    /// one completion queue.
    ///
    /// # Panics
    ///
    /// This method will panic if `count` is 0.
    pub fn cq_count(mut self, count: usize) -> EnvBuilder {
        assert!(count > 0);
        self.cq_count = count;
        self
    }

    /// Set the thread name prefix of each polling thread.
    pub fn name_prefix<S: Into<String>>(mut self, prefix: S) -> EnvBuilder {
        self.name_prefix = Some(prefix.into());
        self
    }

    /// Finalize the [`EnvBuilder`], build the [`Environment`] and initialize the gRPC library.
    pub fn build(self) -> Environment {
        unsafe {
            grpc_sys::grpc_init();
        }
        let mut cqs = Vec::with_capacity(self.cq_count);
        let mut handles = Vec::with_capacity(self.cq_count);
        let (tx, rx) = mpsc::channel();
        for i in 0..self.cq_count {
            let tx_i = tx.clone();
            let mut builder = ThreadBuilder::new();
            if let Some(ref prefix) = self.name_prefix {
                builder = builder.name(format!("{}-{}", prefix, i));
            }
            let handle = builder.spawn(move || poll_queue(tx_i)).unwrap();
            handles.push(handle);
        }
        for _ in 0..self.cq_count {
            cqs.push(rx.recv().unwrap());
        }

        Environment {
            cqs,
            idx: AtomicUsize::new(0),
            _handles: handles,
        }
    }
}

/// An object that used to control concurrency and start gRPC event loop.
pub struct Environment {
    cqs: Vec<CompletionQueue>,
    idx: AtomicUsize,
    _handles: Vec<JoinHandle<()>>,
}

impl Environment {
    /// Initialize gRPC and create a thread pool to poll completion queue. The thread pool size
    /// and the number of completion queue is specified by `cq_count`. Each thread polls one
    /// completion queue.
    ///
    /// # Panics
    ///
    /// This method will panic if `cq_count` is 0.
    pub fn new(cq_count: usize) -> Environment {
        assert!(cq_count > 0);
        EnvBuilder::new()
            .name_prefix("grpc-poll")
            .cq_count(cq_count)
            .build()
    }

    /// Get all the created completion queues.
    pub fn completion_queues(&self) -> &[CompletionQueue] {
        self.cqs.as_slice()
    }

    /// Pick an arbitrary completion queue.
    pub fn pick_cq(&self) -> CompletionQueue {
        let idx = self.idx.fetch_add(1, Ordering::Relaxed);
        self.cqs[idx % self.cqs.len()].clone()
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        for cq in self.completion_queues() {
            // it's safe to shutdown more than once.
            cq.shutdown()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_loop() {
        let mut env = Environment::new(2);

        let q1 = env.pick_cq();
        let q2 = env.pick_cq();
        let q3 = env.pick_cq();
        let cases = vec![(&q1, &q3, true), (&q1, &q2, false)];
        for (lq, rq, is_eq) in cases {
            let lq_ref = lq.borrow().unwrap();
            let rq_ref = rq.borrow().unwrap();
            if is_eq {
                assert_eq!(lq_ref.as_ptr(), rq_ref.as_ptr());
            } else {
                assert_ne!(lq_ref.as_ptr(), rq_ref.as_ptr());
            }
        }

        assert_eq!(env.completion_queues().len(), 2);
        for cq in env.completion_queues() {
            cq.shutdown();
        }

        for handle in env._handles.drain(..) {
            handle.join().unwrap();
        }
    }
}
// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use grpcio_sys::*;
use std::cell::UnsafeCell;
use std::ffi::{c_void, CStr, CString};
use std::fmt::{self, Debug, Formatter};
use std::io::{self, BufRead, Read};
use std::mem::{self, ManuallyDrop, MaybeUninit};

/// A convenient rust wrapper for the type `grpc_slice`.
///
/// It's expected that the slice should be initialized.
#[repr(C)]
pub struct GrpcSlice(grpc_slice);

impl GrpcSlice {
    /// Get the length of the data.
    pub fn len(&self) -> usize {
        unsafe {
            if !self.0.refcount.is_null() {
                self.0.data.refcounted.length
            } else {
                self.0.data.inlined.length as usize
            }
        }
    }

    /// Returns a slice of inner buffer.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            if !self.0.refcount.is_null() {
                let start = self.0.data.refcounted.bytes;
                let len = self.0.data.refcounted.length;
                std::slice::from_raw_parts(start, len)
            } else {
                let len = self.0.data.inlined.length;
                &self.0.data.inlined.bytes[..len as usize]
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a slice from static rust slice.
    ///
    /// Same as `From<&[u8]>` but without copying the buffer.
    #[inline]
    pub fn from_static_slice(s: &'static [u8]) -> GrpcSlice {
        GrpcSlice(unsafe { grpc_slice_from_static_buffer(s.as_ptr() as _, s.len()) })
    }

    /// Creates a `GrpcSlice` from static rust str.
    ///
    /// Same as `from_str` but without allocation.
    #[inline]
    pub fn from_static_str(s: &'static str) -> GrpcSlice {
        GrpcSlice::from_static_slice(s.as_bytes())
    }
}

impl Clone for GrpcSlice {
    /// Clone the slice.
    ///
    /// If the slice is not inlined, the reference count will be increased
    /// instead of copy.
    fn clone(&self) -> Self {
        GrpcSlice(unsafe { grpc_slice_ref(self.0) })
    }
}

impl Default for GrpcSlice {
    /// Returns a default slice, which is empty.
    fn default() -> Self {
        GrpcSlice(unsafe { grpc_empty_slice() })
    }
}

impl Debug for GrpcSlice {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl Drop for GrpcSlice {
    fn drop(&mut self) {
        unsafe {
            grpc_slice_unref(self.0);
        }
    }
}

impl PartialEq<[u8]> for GrpcSlice {
    fn eq(&self, r: &[u8]) -> bool {
        // Technically, the equal function inside vtable should be used.
        // But it's not cheap or safe to create a grpc_slice from rust slice.
        self.as_slice() == r
    }
}

impl PartialEq<GrpcSlice> for GrpcSlice {
    fn eq(&self, r: &GrpcSlice) -> bool {
        unsafe { grpc_slice_eq(self.0, r.0) != 0 }
    }
}

unsafe extern "C" fn drop_vec(ptr: *mut c_void, len: usize) {
    Vec::from_raw_parts(ptr as *mut u8, len, len);
}

impl From<Vec<u8>> for GrpcSlice {
    /// Converts a `Vec<u8>` into `GrpcSlice`.
    ///
    /// If v can't fit inline, there will be allocations.
    #[inline]
    fn from(mut v: Vec<u8>) -> GrpcSlice {
        if v.is_empty() {
            return GrpcSlice::default();
        }

        if v.len() == v.capacity() {
            let slice = unsafe {
                grpcio_sys::grpc_slice_new_with_len(v.as_mut_ptr() as _, v.len(), Some(drop_vec))
            };
            mem::forget(v);
            return GrpcSlice(slice);
        }

        unsafe {
            GrpcSlice(grpcio_sys::grpc_slice_from_copied_buffer(
                v.as_mut_ptr() as _,
                v.len(),
            ))
        }
    }
}

/// Creates a `GrpcSlice` from rust string.
///
/// If the string can't fit inline, there will be allocations.
impl From<String> for GrpcSlice {
    #[inline]
    fn from(s: String) -> GrpcSlice {
        GrpcSlice::from(s.into_bytes())
    }
}

/// Creates a `GrpcSlice` from rust cstring.
///
/// If the cstring can't fit inline, there will be allocations.
impl From<CString> for GrpcSlice {
    #[inline]
    fn from(s: CString) -> GrpcSlice {
        GrpcSlice::from(s.into_bytes())
    }
}

/// Creates a `GrpcSlice` from rust slice.
///
/// The data inside slice will be cloned. If the data can't fit inline,
/// necessary buffer will be allocated.
impl From<&'_ [u8]> for GrpcSlice {
    #[inline]
    fn from(s: &'_ [u8]) -> GrpcSlice {
        GrpcSlice(unsafe { grpc_slice_from_copied_buffer(s.as_ptr() as _, s.len()) })
    }
}

/// Creates a `GrpcSlice` from rust str.
///
/// The data inside str will be cloned. If the data can't fit inline,
/// necessary buffer will be allocated.
impl From<&'_ str> for GrpcSlice {
    #[inline]
    fn from(s: &'_ str) -> GrpcSlice {
        GrpcSlice::from(s.as_bytes())
    }
}

/// Creates a `GrpcSlice` from rust `CStr`.
///
/// The data inside `CStr` will be cloned. If the data can't fit inline,
/// necessary buffer will be allocated.
impl From<&'_ CStr> for GrpcSlice {
    #[inline]
    fn from(s: &'_ CStr) -> GrpcSlice {
        GrpcSlice::from(s.to_bytes())
    }
}

/// A collection of `GrpcBytes`.
#[repr(C)]
pub struct GrpcByteBuffer(*mut grpc_byte_buffer);

impl GrpcByteBuffer {
    #[inline]
    pub unsafe fn from_raw(ptr: *mut grpc_byte_buffer) -> GrpcByteBuffer {
        GrpcByteBuffer(ptr)
    }
}

impl<'a> From<&'a [GrpcSlice]> for GrpcByteBuffer {
    /// Create a buffer from the given slice array.
    ///
    /// A buffer is allocated for the whole slice array, and every slice will
    /// be `Clone::clone` into the buffer.
    fn from(slice: &'a [GrpcSlice]) -> Self {
        let len = slice.len();
        unsafe {
            let s = slice.as_ptr() as *const grpc_slice as *const UnsafeCell<grpc_slice>;
            // hack: see From<&GrpcSlice>.
            GrpcByteBuffer(grpc_raw_byte_buffer_create((*s).get(), len))
        }
    }
}

impl<'a> From<&'a GrpcSlice> for GrpcByteBuffer {
    /// Create a buffer from the given single slice.
    ///
    /// A buffer, which length is 1, is allocated for the slice.
    #[allow(clippy::cast_ref_to_mut)]
    fn from(s: &'a GrpcSlice) -> GrpcByteBuffer {
        unsafe {
            // hack: buffer_create accepts an mutable pointer to indicate it mutate
            // ref count. Ref count is recorded by atomic variable, which is considered
            // `Sync` in rust. This is an interesting difference in what is *mutable*
            // between C++ and rust.
            // Using `UnsafeCell` to avoid raw cast, which is UB.
            let s = &*(s as *const GrpcSlice as *const grpc_slice as *const UnsafeCell<grpc_slice>);
            GrpcByteBuffer(grpc_raw_byte_buffer_create((*s).get(), 1))
        }
    }
}

impl Clone for GrpcByteBuffer {
    fn clone(&self) -> Self {
        unsafe { GrpcByteBuffer(grpc_byte_buffer_copy(self.0)) }
    }
}

impl Drop for GrpcByteBuffer {
    fn drop(&mut self) {
        unsafe { grpc_byte_buffer_destroy(self.0) }
    }
}

/// A zero-copy reader for the message payload.
///
/// To achieve zero-copy, use the BufRead API `fill_buf` and `consume`
/// to operate the reader.
#[repr(C)]
pub struct GrpcByteBufferReader {
    reader: grpc_byte_buffer_reader,
    /// Current reading buffer.
    // This is a temporary buffer that may need to be dropped before every
    // iteration. So use `ManuallyDrop` to control the behavior more clean
    // and precisely.
    slice: ManuallyDrop<GrpcSlice>,
    /// The offset of `slice` that has not been read.
    offset: usize,
    /// How many bytes pending for reading.
    remain: usize,
}

impl GrpcByteBufferReader {
    /// Creates a reader for the `GrpcByteBuffer`.
    ///
    /// `buf` is stored inside the reader, and dropped when the reader is dropped.
    pub fn new(buf: GrpcByteBuffer) -> GrpcByteBufferReader {
        let mut reader = MaybeUninit::uninit();
        let mut s = MaybeUninit::uninit();
        unsafe {
            let code = grpc_byte_buffer_reader_init(reader.as_mut_ptr(), buf.0);
            assert_eq!(code, 1);
            if 0 == grpc_byte_buffer_reader_next(reader.as_mut_ptr(), s.as_mut_ptr()) {
                s.as_mut_ptr().write(grpc_empty_slice());
            }
            let remain = grpc_byte_buffer_length((*reader.as_mut_ptr()).buffer_out);
            // buf is stored inside `reader` as `buffer_in`, so do not drop it.
            mem::forget(buf);

            GrpcByteBufferReader {
                reader: reader.assume_init(),
                slice: ManuallyDrop::new(GrpcSlice(s.assume_init())),
                offset: 0,
                remain,
            }
        }
    }

    /// Get the next slice from reader.
    fn load_next_slice(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.slice);
            if 0 == grpc_byte_buffer_reader_next(&mut self.reader, &mut self.slice.0) {
                self.slice = ManuallyDrop::new(GrpcSlice::default());
            }
        }
        self.offset = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.remain
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.remain == 0
    }
}

impl Read for GrpcByteBufferReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.fill_buf()?.read(buf)?;
        self.consume(read);
        Ok(read)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let cap = self.remain;
        buf.reserve(cap);
        let old_len = buf.len();
        while self.remain > 0 {
            let read = {
                let s = match self.fill_buf() {
                    Ok(s) => s,
                    Err(e) => {
                        unsafe {
                            buf.set_len(old_len);
                        }
                        return Err(e);
                    }
                };
                buf.extend_from_slice(s);
                s.len()
            };
            self.consume(read);
        }
        Ok(cap)
    }
}

impl BufRead for GrpcByteBufferReader {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.slice.is_empty() {
            return Ok(&[]);
        }
        Ok(unsafe { self.slice.as_slice().get_unchecked(self.offset..) })
    }

    fn consume(&mut self, mut amt: usize) {
        if amt > self.remain {
            amt = self.remain;
        }
        self.remain -= amt;
        let mut offset = self.offset + amt;
        while offset >= self.slice.len() && offset > 0 {
            offset -= self.slice.len();
            self.load_next_slice();
        }
        self.offset = offset;
    }
}

impl Drop for GrpcByteBufferReader {
    fn drop(&mut self) {
        unsafe {
            grpc_byte_buffer_reader_destroy(&mut self.reader);
            ManuallyDrop::drop(&mut self.slice);
            grpc_byte_buffer_destroy(self.reader.buffer_in);
        }
    }
}

unsafe impl Sync for GrpcByteBufferReader {}
unsafe impl Send for GrpcByteBufferReader {}

#[cfg(feature = "prost-codec")]
impl bytes::Buf for GrpcByteBufferReader {
    fn remaining(&self) -> usize {
        self.remain
    }

    fn bytes(&self) -> &[u8] {
        // This is similar but not identical to `BuffRead::fill_buf`, since `self`
        // is not mutable, we can only return bytes up to the end of the current
        // slice.

        // Optimization for empty slice
        if self.slice.is_empty() {
            return &[];
        }

        unsafe { self.slice.as_slice().get_unchecked(self.offset..) }
    }

    fn advance(&mut self, cnt: usize) {
        self.consume(cnt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_message_reader(seed: Vec<u8>, copy_count: usize) -> GrpcByteBufferReader {
        let data = vec![GrpcSlice::from(seed); copy_count];
        let buf = GrpcByteBuffer::from(data.as_slice());
        GrpcByteBufferReader::new(buf)
    }

    #[test]
    fn test_grpc_slice() {
        let empty = GrpcSlice::default();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(empty.as_slice().is_empty());

        let a = vec![0, 2, 1, 3, 8];
        let slice = GrpcSlice::from(a.clone());
        assert_eq!(a.as_slice(), slice.as_slice());
        assert_eq!(a.len(), slice.len());
        assert_eq!(&slice, &*a);

        let a = vec![5; 64];
        let slice = GrpcSlice::from(a.clone());
        assert_eq!(a.as_slice(), slice.as_slice());
        assert_eq!(a.len(), slice.len());
        assert_eq!(&slice, &*a);

        let a = vec![];
        let slice = GrpcSlice::from(a);
        assert_eq!(empty, slice);
    }

    #[test]
    // Old code crashes under a very weird circumstance, due to a typo in `MessageReader::consume`
    fn test_typo_len_offset() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // half of the size of `data`
        let half_size = data.len() / 2;
        let slice = GrpcSlice::from(data.clone());
        let buffer = GrpcByteBuffer::from(&slice);
        let mut reader = GrpcByteBufferReader::new(buffer);
        assert_eq!(reader.len(), data.len());
        // first 3 elements of `data`
        let mut buf = vec![0; half_size];
        reader.read(buf.as_mut_slice()).unwrap();
        assert_eq!(data[..half_size], *buf.as_slice());
        assert_eq!(reader.len(), data.len() - half_size);
        assert!(!reader.is_empty());
        reader.read(&mut buf).unwrap();
        assert_eq!(data[half_size..], *buf.as_slice());
        assert_eq!(reader.len(), 0);
        assert!(reader.is_empty());
    }

    #[test]
    fn test_message_reader() {
        for len in 0..=1024 {
            for n_slice in 1..=4 {
                let source = vec![len as u8; len];
                let expect = vec![len as u8; len * n_slice];
                // Test read.
                let mut reader = new_message_reader(source.clone(), n_slice);
                let mut dest = [0; 7];
                let amt = reader.read(&mut dest).unwrap();

                assert_eq!(
                    dest[..amt],
                    expect[..amt],
                    "len: {}, nslice: {}",
                    len,
                    n_slice
                );

                // Read after move.
                let mut box_reader = Box::new(reader);
                let amt = box_reader.read(&mut dest).unwrap();
                assert_eq!(
                    dest[..amt],
                    expect[..amt],
                    "len: {}, nslice: {}",
                    len,
                    n_slice
                );

                // Test read_to_end.
                let mut reader = new_message_reader(source.clone(), n_slice);
                let mut dest = vec![];
                reader.read_to_end(&mut dest).unwrap();
                assert_eq!(dest, expect, "len: {}, nslice: {}", len, n_slice);

                assert_eq!(0, reader.len());
                assert_eq!(0, reader.read(&mut [1]).unwrap());

                // Test arbitrary consuming.
                let mut reader = new_message_reader(source.clone(), n_slice);
                reader.consume(source.len() * (n_slice - 1));
                let mut dest = vec![];
                reader.read_to_end(&mut dest).unwrap();
                assert_eq!(
                    dest.len(),
                    source.len(),
                    "len: {}, nslice: {}",
                    len,
                    n_slice
                );
                assert_eq!(
                    *dest,
                    expect[expect.len() - source.len()..],
                    "len: {}, nslice: {}",
                    len,
                    n_slice
                );
                assert_eq!(0, reader.len());
                assert_eq!(0, reader.read(&mut [1]).unwrap());
            }
        }
    }

    #[test]
    fn test_converter() {
        let a = vec![1, 2, 3, 0];
        assert_eq!(GrpcSlice::from(a.clone()).as_slice(), a.as_slice());
        assert_eq!(GrpcSlice::from(a.as_slice()).as_slice(), a.as_slice());

        let s = "abcd".to_owned();
        assert_eq!(GrpcSlice::from(s.clone()).as_slice(), s.as_bytes());
        assert_eq!(GrpcSlice::from(s.as_str()).as_slice(), s.as_bytes());

        let cs = CString::new(s.clone()).unwrap();
        assert_eq!(GrpcSlice::from(cs.clone()).as_slice(), s.as_bytes());
        assert_eq!(GrpcSlice::from(cs.as_c_str()).as_slice(), s.as_bytes());
    }

    #[cfg(feature = "prost-codec")]
    #[test]
    fn test_buf_impl() {
        use bytes::Buf;

        for len in 0..1024 + 1 {
            for n_slice in 1..4 {
                let source = vec![len as u8; len];

                let mut reader = new_message_reader(source.clone(), n_slice);

                let mut remaining = len * n_slice;
                let mut count = 100;
                while reader.remaining() > 0 {
                    assert_eq!(remaining, reader.remaining());
                    let bytes = Buf::bytes(&reader);
                    bytes.iter().for_each(|b| assert_eq!(*b, len as u8));
                    let mut read = bytes.len();
                    // We don't have to advance by the whole amount we read.
                    if read > 5 && len % 2 == 0 {
                        read -= 5;
                    }
                    reader.advance(read);
                    remaining -= read;
                    count -= 1;
                    assert!(count > 0);
                }

                assert_eq!(0, remaining);
                assert_eq!(0, reader.remaining());
            }
        }
    }
}
// This module defines a common API for caching internal runtime state.
// The `thread_local` crate provides an extremely optimized version of this.
// However, if the perf-cache feature is disabled, then we drop the
// thread_local dependency and instead use a pretty naive caching mechanism
// with a mutex.
//
// Strictly speaking, the CachedGuard isn't necessary for the much more
// flexible thread_local API, but implementing thread_local's API doesn't
// seem possible in purely safe code.

pub use self::imp::{Cached, CachedGuard};

#[cfg(feature = "perf-cache")]
mod imp {
    use thread_local::CachedThreadLocal;

    #[derive(Debug)]
    pub struct Cached<T: Send>(CachedThreadLocal<T>);

    #[derive(Debug)]
    pub struct CachedGuard<'a, T: 'a>(&'a T);

    impl<T: Send> Cached<T> {
        pub fn new() -> Cached<T> {
            Cached(CachedThreadLocal::new())
        }

        pub fn get_or(&self, create: impl FnOnce() -> T) -> CachedGuard<T> {
            CachedGuard(self.0.get_or(|| create()))
        }
    }

    impl<'a, T: Send> CachedGuard<'a, T> {
        pub fn value(&self) -> &T {
            self.0
        }
    }
}

#[cfg(not(feature = "perf-cache"))]
mod imp {
    use std::marker::PhantomData;
    use std::panic::UnwindSafe;
    use std::sync::Mutex;

    #[derive(Debug)]
    pub struct Cached<T: Send> {
        stack: Mutex<Vec<T>>,
        /// When perf-cache is enabled, the thread_local crate is used, and
        /// its CachedThreadLocal impls Send, Sync and UnwindSafe, but NOT
        /// RefUnwindSafe. However, a Mutex impls RefUnwindSafe. So in order
        /// to keep the APIs consistent regardless of whether perf-cache is
        /// enabled, we force this type to NOT impl RefUnwindSafe too.
        ///
        /// Ideally, we should always impl RefUnwindSafe, but it seems a little
        /// tricky to do that right now.
        ///
        /// See also: https://github.com/rust-lang/regex/issues/576
        _phantom: PhantomData<Box<dyn Send + Sync + UnwindSafe>>,
    }

    #[derive(Debug)]
    pub struct CachedGuard<'a, T: 'a + Send> {
        cache: &'a Cached<T>,
        value: Option<T>,
    }

    impl<T: Send> Cached<T> {
        pub fn new() -> Cached<T> {
            Cached { stack: Mutex::new(vec![]), _phantom: PhantomData }
        }

        pub fn get_or(&self, create: impl FnOnce() -> T) -> CachedGuard<T> {
            let mut stack = self.stack.lock().unwrap();
            match stack.pop() {
                None => CachedGuard { cache: self, value: Some(create()) },
                Some(value) => CachedGuard { cache: self, value: Some(value) },
            }
        }

        fn put(&self, value: T) {
            let mut stack = self.stack.lock().unwrap();
            stack.push(value);
        }
    }

    impl<'a, T: Send> CachedGuard<'a, T> {
        pub fn value(&self) -> &T {
            self.value.as_ref().unwrap()
        }
    }

    impl<'a, T: Send> Drop for CachedGuard<'a, T> {
        fn drop(&mut self) {
            if let Some(value) = self.value.take() {
                self.cache.put(value);
            }
        }
    }
}
use std::fmt;
use std::iter::repeat;

/// An error that occurred during parsing or compiling a regular expression.
#[derive(Clone, PartialEq)]
pub enum Error {
    /// A syntax error.
    Syntax(String),
    /// The compiled program exceeded the set size limit.
    /// The argument is the size limit imposed.
    CompiledTooBig(usize),
    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
}

impl ::std::error::Error for Error {
    // TODO: Remove this method entirely on the next breaking semver release.
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match *self {
            Error::Syntax(ref err) => err,
            Error::CompiledTooBig(_) => "compiled program too big",
            Error::__Nonexhaustive => unreachable!(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Syntax(ref err) => err.fmt(f),
            Error::CompiledTooBig(limit) => write!(
                f,
                "Compiled regex exceeds size limit of {} bytes.",
                limit
            ),
            Error::__Nonexhaustive => unreachable!(),
        }
    }
}

// We implement our own Debug implementation so that we show nicer syntax
// errors when people use `Regex::new(...).unwrap()`. It's a little weird,
// but the `Syntax` variant is already storing a `String` anyway, so we might
// as well format it nicely.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Syntax(ref err) => {
                let hr: String = repeat('~').take(79).collect();
                writeln!(f, "Syntax(")?;
                writeln!(f, "{}", hr)?;
                writeln!(f, "{}", err)?;
                writeln!(f, "{}", hr)?;
                write!(f, ")")?;
                Ok(())
            }
            Error::CompiledTooBig(limit) => {
                f.debug_tuple("CompiledTooBig").field(&limit).finish()
            }
            Error::__Nonexhaustive => {
                f.debug_tuple("__Nonexhaustive").finish()
            }
        }
    }
}
// NOTE: The following code was generated by "scripts/frequencies.py", do not
// edit directly

pub const BYTE_FREQUENCIES: [u8; 256] = [
    55,  // '\x00'
    52,  // '\x01'
    51,  // '\x02'
    50,  // '\x03'
    49,  // '\x04'
    48,  // '\x05'
    47,  // '\x06'
    46,  // '\x07'
    45,  // '\x08'
    103, // '\t'
    242, // '\n'
    66,  // '\x0b'
    67,  // '\x0c'
    229, // '\r'
    44,  // '\x0e'
    43,  // '\x0f'
    42,  // '\x10'
    41,  // '\x11'
    40,  // '\x12'
    39,  // '\x13'
    38,  // '\x14'
    37,  // '\x15'
    36,  // '\x16'
    35,  // '\x17'
    34,  // '\x18'
    33,  // '\x19'
    56,  // '\x1a'
    32,  // '\x1b'
    31,  // '\x1c'
    30,  // '\x1d'
    29,  // '\x1e'
    28,  // '\x1f'
    255, // ' '
    148, // '!'
    164, // '"'
    149, // '#'
    136, // '$'
    160, // '%'
    155, // '&'
    173, // "'"
    221, // '('
    222, // ')'
    134, // '*'
    122, // '+'
    232, // ','
    202, // '-'
    215, // '.'
    224, // '/'
    208, // '0'
    220, // '1'
    204, // '2'
    187, // '3'
    183, // '4'
    179, // '5'
    177, // '6'
    168, // '7'
    178, // '8'
    200, // '9'
    226, // ':'
    195, // ';'
    154, // '<'
    184, // '='
    174, // '>'
    126, // '?'
    120, // '@'
    191, // 'A'
    157, // 'B'
    194, // 'C'
    170, // 'D'
    189, // 'E'
    162, // 'F'
    161, // 'G'
    150, // 'H'
    193, // 'I'
    142, // 'J'
    137, // 'K'
    171, // 'L'
    176, // 'M'
    185, // 'N'
    167, // 'O'
    186, // 'P'
    112, // 'Q'
    175, // 'R'
    192, // 'S'
    188, // 'T'
    156, // 'U'
    140, // 'V'
    143, // 'W'
    123, // 'X'
    133, // 'Y'
    128, // 'Z'
    147, // '['
    138, // '\\'
    146, // ']'
    114, // '^'
    223, // '_'
    151, // '`'
    249, // 'a'
    216, // 'b'
    238, // 'c'
    236, // 'd'
    253, // 'e'
    227, // 'f'
    218, // 'g'
    230, // 'h'
    247, // 'i'
    135, // 'j'
    180, // 'k'
    241, // 'l'
    233, // 'm'
    246, // 'n'
    244, // 'o'
    231, // 'p'
    139, // 'q'
    245, // 'r'
    243, // 's'
    251, // 't'
    235, // 'u'
    201, // 'v'
    196, // 'w'
    240, // 'x'
    214, // 'y'
    152, // 'z'
    182, // '{'
    205, // '|'
    181, // '}'
    127, // '~'
    27,  // '\x7f'
    212, // '\x80'
    211, // '\x81'
    210, // '\x82'
    213, // '\x83'
    228, // '\x84'
    197, // '\x85'
    169, // '\x86'
    159, // '\x87'
    131, // '\x88'
    172, // '\x89'
    105, // '\x8a'
    80,  // '\x8b'
    98,  // '\x8c'
    96,  // '\x8d'
    97,  // '\x8e'
    81,  // '\x8f'
    207, // '\x90'
    145, // '\x91'
    116, // '\x92'
    115, // '\x93'
    144, // '\x94'
    130, // '\x95'
    153, // '\x96'
    121, // '\x97'
    107, // '\x98'
    132, // '\x99'
    109, // '\x9a'
    110, // '\x9b'
    124, // '\x9c'
    111, // '\x9d'
    82,  // '\x9e'
    108, // '\x9f'
    118, // '\xa0'
    141, // '¡'
    113, // '¢'
    129, // '£'
    119, // '¤'
    125, // '¥'
    165, // '¦'
    117, // '§'
    92,  // '¨'
    106, // '©'
    83,  // 'ª'
    72,  // '«'
    99,  // '¬'
    93,  // '\xad'
    65,  // '®'
    79,  // '¯'
    166, // '°'
    237, // '±'
    163, // '²'
    199, // '³'
    190, // '´'
    225, // 'µ'
    209, // '¶'
    203, // '·'
    198, // '¸'
    217, // '¹'
    219, // 'º'
    206, // '»'
    234, // '¼'
    248, // '½'
    158, // '¾'
    239, // '¿'
    255, // 'À'
    255, // 'Á'
    255, // 'Â'
    255, // 'Ã'
    255, // 'Ä'
    255, // 'Å'
    255, // 'Æ'
    255, // 'Ç'
    255, // 'È'
    255, // 'É'
    255, // 'Ê'
    255, // 'Ë'
    255, // 'Ì'
    255, // 'Í'
    255, // 'Î'
    255, // 'Ï'
    255, // 'Ð'
    255, // 'Ñ'
    255, // 'Ò'
    255, // 'Ó'
    255, // 'Ô'
    255, // 'Õ'
    255, // 'Ö'
    255, // '×'
    255, // 'Ø'
    255, // 'Ù'
    255, // 'Ú'
    255, // 'Û'
    255, // 'Ü'
    255, // 'Ý'
    255, // 'Þ'
    255, // 'ß'
    255, // 'à'
    255, // 'á'
    255, // 'â'
    255, // 'ã'
    255, // 'ä'
    255, // 'å'
    255, // 'æ'
    255, // 'ç'
    255, // 'è'
    255, // 'é'
    255, // 'ê'
    255, // 'ë'
    255, // 'ì'
    255, // 'í'
    255, // 'î'
    255, // 'ï'
    255, // 'ð'
    255, // 'ñ'
    255, // 'ò'
    255, // 'ó'
    255, // 'ô'
    255, // 'õ'
    255, // 'ö'
    255, // '÷'
    255, // 'ø'
    255, // 'ù'
    255, // 'ú'
    255, // 'û'
    255, // 'ü'
    255, // 'ý'
    255, // 'þ'
    255, // 'ÿ'
];
use std::char;
use std::cmp::Ordering;
use std::fmt;
use std::ops;
use std::u32;

use syntax;

use literal::LiteralSearcher;
use prog::InstEmptyLook;
use utf8::{decode_last_utf8, decode_utf8};

/// Represents a location in the input.
#[derive(Clone, Copy, Debug)]
pub struct InputAt {
    pos: usize,
    c: Char,
    byte: Option<u8>,
    len: usize,
}

impl InputAt {
    /// Returns true iff this position is at the beginning of the input.
    pub fn is_start(&self) -> bool {
        self.pos == 0
    }

    /// Returns true iff this position is past the end of the input.
    pub fn is_end(&self) -> bool {
        self.c.is_none() && self.byte.is_none()
    }

    /// Returns the character at this position.
    ///
    /// If this position is just before or after the input, then an absent
    /// character is returned.
    pub fn char(&self) -> Char {
        self.c
    }

    /// Returns the byte at this position.
    pub fn byte(&self) -> Option<u8> {
        self.byte
    }

    /// Returns the UTF-8 width of the character at this position.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the UTF-8 width of the character at this position
    /// is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the byte offset of this position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Returns the byte offset of the next position in the input.
    pub fn next_pos(&self) -> usize {
        self.pos + self.len
    }
}

/// An abstraction over input used in the matching engines.
pub trait Input: fmt::Debug {
    /// Return an encoding of the position at byte offset `i`.
    fn at(&self, i: usize) -> InputAt;

    /// Return the Unicode character occurring next to `at`.
    ///
    /// If no such character could be decoded, then `Char` is absent.
    fn next_char(&self, at: InputAt) -> Char;

    /// Return the Unicode character occurring previous to `at`.
    ///
    /// If no such character could be decoded, then `Char` is absent.
    fn previous_char(&self, at: InputAt) -> Char;

    /// Return true if the given empty width instruction matches at the
    /// input position given.
    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool;

    /// Scan the input for a matching prefix.
    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt>;

    /// The number of bytes in the input.
    fn len(&self) -> usize;

    /// Whether the input is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the given input as a sequence of bytes.
    fn as_bytes(&self) -> &[u8];
}

impl<'a, T: Input> Input for &'a T {
    fn at(&self, i: usize) -> InputAt {
        (**self).at(i)
    }

    fn next_char(&self, at: InputAt) -> Char {
        (**self).next_char(at)
    }

    fn previous_char(&self, at: InputAt) -> Char {
        (**self).previous_char(at)
    }

    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool {
        (**self).is_empty_match(at, empty)
    }

    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt> {
        (**self).prefix_at(prefixes, at)
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn as_bytes(&self) -> &[u8] {
        (**self).as_bytes()
    }
}

/// An input reader over characters.
#[derive(Clone, Copy, Debug)]
pub struct CharInput<'t>(&'t [u8]);

impl<'t> CharInput<'t> {
    /// Return a new character input reader for the given string.
    pub fn new(s: &'t [u8]) -> CharInput<'t> {
        CharInput(s)
    }
}

impl<'t> ops::Deref for CharInput<'t> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0
    }
}

impl<'t> Input for CharInput<'t> {
    fn at(&self, i: usize) -> InputAt {
        if i >= self.len() {
            InputAt { pos: self.len(), c: None.into(), byte: None, len: 0 }
        } else {
            let c = decode_utf8(&self[i..]).map(|(c, _)| c).into();
            InputAt { pos: i, c: c, byte: None, len: c.len_utf8() }
        }
    }

    fn next_char(&self, at: InputAt) -> Char {
        at.char()
    }

    fn previous_char(&self, at: InputAt) -> Char {
        decode_last_utf8(&self[..at.pos()]).map(|(c, _)| c).into()
    }

    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool {
        use prog::EmptyLook::*;
        match empty.look {
            StartLine => {
                let c = self.previous_char(at);
                at.pos() == 0 || c == '\n'
            }
            EndLine => {
                let c = self.next_char(at);
                at.pos() == self.len() || c == '\n'
            }
            StartText => at.pos() == 0,
            EndText => at.pos() == self.len(),
            WordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() != c2.is_word_char()
            }
            NotWordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() == c2.is_word_char()
            }
            WordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_byte() != c2.is_word_byte()
            }
            NotWordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_byte() == c2.is_word_byte()
            }
        }
    }

    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt> {
        prefixes.find(&self[at.pos()..]).map(|(s, _)| self.at(at.pos() + s))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_bytes(&self) -> &[u8] {
        self.0
    }
}

/// An input reader over bytes.
#[derive(Clone, Copy, Debug)]
pub struct ByteInput<'t> {
    text: &'t [u8],
    only_utf8: bool,
}

impl<'t> ByteInput<'t> {
    /// Return a new byte-based input reader for the given string.
    pub fn new(text: &'t [u8], only_utf8: bool) -> ByteInput<'t> {
        ByteInput { text: text, only_utf8: only_utf8 }
    }
}

impl<'t> ops::Deref for ByteInput<'t> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.text
    }
}

impl<'t> Input for ByteInput<'t> {
    fn at(&self, i: usize) -> InputAt {
        if i >= self.len() {
            InputAt { pos: self.len(), c: None.into(), byte: None, len: 0 }
        } else {
            InputAt {
                pos: i,
                c: None.into(),
                byte: self.get(i).cloned(),
                len: 1,
            }
        }
    }

    fn next_char(&self, at: InputAt) -> Char {
        decode_utf8(&self[at.pos()..]).map(|(c, _)| c).into()
    }

    fn previous_char(&self, at: InputAt) -> Char {
        decode_last_utf8(&self[..at.pos()]).map(|(c, _)| c).into()
    }

    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool {
        use prog::EmptyLook::*;
        match empty.look {
            StartLine => {
                let c = self.previous_char(at);
                at.pos() == 0 || c == '\n'
            }
            EndLine => {
                let c = self.next_char(at);
                at.pos() == self.len() || c == '\n'
            }
            StartText => at.pos() == 0,
            EndText => at.pos() == self.len(),
            WordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() != c2.is_word_char()
            }
            NotWordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() == c2.is_word_char()
            }
            WordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                if self.only_utf8 {
                    // If we must match UTF-8, then we can't match word
                    // boundaries at invalid UTF-8.
                    if c1.is_none() && !at.is_start() {
                        return false;
                    }
                    if c2.is_none() && !at.is_end() {
                        return false;
                    }
                }
                c1.is_word_byte() != c2.is_word_byte()
            }
            NotWordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                if self.only_utf8 {
                    // If we must match UTF-8, then we can't match word
                    // boundaries at invalid UTF-8.
                    if c1.is_none() && !at.is_start() {
                        return false;
                    }
                    if c2.is_none() && !at.is_end() {
                        return false;
                    }
                }
                c1.is_word_byte() == c2.is_word_byte()
            }
        }
    }

    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt> {
        prefixes.find(&self[at.pos()..]).map(|(s, _)| self.at(at.pos() + s))
    }

    fn len(&self) -> usize {
        self.text.len()
    }

    fn as_bytes(&self) -> &[u8] {
        self.text
    }
}

/// An inline representation of `Option<char>`.
///
/// This eliminates the need to do case analysis on `Option<char>` to determine
/// ordinality with other characters.
///
/// (The `Option<char>` is not related to encoding. Instead, it is used in the
/// matching engines to represent the beginning and ending boundaries of the
/// search text.)
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Char(u32);

impl fmt::Debug for Char {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match char::from_u32(self.0) {
            None => write!(f, "Empty"),
            Some(c) => write!(f, "{:?}", c),
        }
    }
}

impl Char {
    /// Returns true iff the character is absent.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns the length of the character's UTF-8 encoding.
    ///
    /// If the character is absent, then `1` is returned.
    #[inline]
    pub fn len_utf8(self) -> usize {
        char::from_u32(self.0).map_or(1, |c| c.len_utf8())
    }

    /// Returns true iff the character is a word character.
    ///
    /// If the character is absent, then false is returned.
    pub fn is_word_char(self) -> bool {
        // is_word_character can panic if the Unicode data for \w isn't
        // available. However, our compiler ensures that if a Unicode word
        // boundary is used, then the data must also be available. If it isn't,
        // then the compiler returns an error.
        char::from_u32(self.0).map_or(false, syntax::is_word_character)
    }

    /// Returns true iff the byte is a word byte.
    ///
    /// If the byte is absent, then false is returned.
    pub fn is_word_byte(self) -> bool {
        match char::from_u32(self.0) {
            Some(c) if c <= '\u{7F}' => syntax::is_word_byte(c as u8),
            None | Some(_) => false,
        }
    }
}

impl From<char> for Char {
    fn from(c: char) -> Char {
        Char(c as u32)
    }
}

impl From<Option<char>> for Char {
    fn from(c: Option<char>) -> Char {
        c.map_or(Char(u32::MAX), |c| c.into())
    }
}

impl PartialEq<char> for Char {
    #[inline]
    fn eq(&self, other: &char) -> bool {
        self.0 == *other as u32
    }
}

impl PartialEq<Char> for char {
    #[inline]
    fn eq(&self, other: &Char) -> bool {
        *self as u32 == other.0
    }
}

impl PartialOrd<char> for Char {
    #[inline]
    fn partial_cmp(&self, other: &char) -> Option<Ordering> {
        self.0.partial_cmp(&(*other as u32))
    }
}

impl PartialOrd<Char> for char {
    #[inline]
    fn partial_cmp(&self, other: &Char) -> Option<Ordering> {
        (*self as u32).partial_cmp(&other.0)
    }
}
#[test]
fn empty_regex_empty_match() {
    let re = regex!("");
    assert_eq!(vec![(0, 0)], findall!(re, ""));
}

#[test]
fn empty_regex_nonempty_match() {
    let re = regex!("");
    assert_eq!(vec![(0, 0), (1, 1), (2, 2), (3, 3)], findall!(re, "abc"));
}

#[test]
fn one_zero_length_match() {
    let re = regex!(r"[0-9]*");
    assert_eq!(vec![(0, 0), (1, 2), (3, 4)], findall!(re, "a1b2"));
}

#[test]
fn many_zero_length_match() {
    let re = regex!(r"[0-9]*");
    assert_eq!(
        vec![(0, 0), (1, 2), (3, 3), (4, 4), (5, 6)],
        findall!(re, "a1bbb2")
    );
}

#[test]
fn many_sequential_zero_length_match() {
    let re = regex!(r"[0-9]?");
    assert_eq!(
        vec![(0, 0), (1, 2), (2, 3), (4, 5), (6, 6)],
        findall!(re, "a12b3c")
    );
}

#[test]
fn quoted_bracket_set() {
    let re = regex!(r"([\x{5b}\x{5d}])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
    let re = regex!(r"([\[\]])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
}

#[test]
fn first_range_starts_with_left_bracket() {
    let re = regex!(r"([\[-z])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
}

#[test]
fn range_ends_with_escape() {
    let re = regex!(r"([\[-\x{5d}])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
}

#[test]
fn empty_match_find_iter() {
    let re = regex!(r".*?");
    assert_eq!(vec![(0, 0), (1, 1), (2, 2), (3, 3)], findall!(re, "abc"));
}

#[test]
fn empty_match_captures_iter() {
    let re = regex!(r".*?");
    let ms: Vec<_> = re
        .captures_iter(text!("abc"))
        .map(|c| c.get(0).unwrap())
        .map(|m| (m.start(), m.end()))
        .collect();
    assert_eq!(ms, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
}

#[test]
fn capture_names() {
    let re = regex!(r"(.)(?P<a>.)");
    assert_eq!(3, re.captures_len());
    assert_eq!((3, Some(3)), re.capture_names().size_hint());
    assert_eq!(
        vec![None, None, Some("a")],
        re.capture_names().collect::<Vec<_>>()
    );
}

#[test]
fn regex_string() {
    assert_eq!(r"[a-zA-Z0-9]+", regex!(r"[a-zA-Z0-9]+").as_str());
    assert_eq!(r"[a-zA-Z0-9]+", &format!("{}", regex!(r"[a-zA-Z0-9]+")));
    assert_eq!(r"[a-zA-Z0-9]+", &format!("{:?}", regex!(r"[a-zA-Z0-9]+")));
}

#[test]
fn capture_index() {
    let re = regex!(r"^(?P<name>.+)$");
    let cap = re.captures(t!("abc")).unwrap();
    assert_eq!(&cap[0], t!("abc"));
    assert_eq!(&cap[1], t!("abc"));
    assert_eq!(&cap["name"], t!("abc"));
}

#[test]
#[should_panic]
#[cfg_attr(all(target_env = "msvc", target_pointer_width = "32"), ignore)]
fn capture_index_panic_usize() {
    let re = regex!(r"^(?P<name>.+)$");
    let cap = re.captures(t!("abc")).unwrap();
    let _ = cap[2];
}

#[test]
#[should_panic]
#[cfg_attr(all(target_env = "msvc", target_pointer_width = "32"), ignore)]
fn capture_index_panic_name() {
    let re = regex!(r"^(?P<name>.+)$");
    let cap = re.captures(t!("abc")).unwrap();
    let _ = cap["bad name"];
}

#[test]
fn capture_index_lifetime() {
    // This is a test of whether the types on `caps["..."]` are general
    // enough. If not, this will fail to typecheck.
    fn inner(s: &str) -> usize {
        let re = regex!(r"(?P<number>[0-9]+)");
        let caps = re.captures(t!(s)).unwrap();
        caps["number"].len()
    }
    assert_eq!(3, inner("123"));
}

#[test]
fn capture_misc() {
    let re = regex!(r"(.)(?P<a>a)?(.)(?P<b>.)");
    let cap = re.captures(t!("abc")).unwrap();

    assert_eq!(5, cap.len());

    assert_eq!((0, 3), {
        let m = cap.get(0).unwrap();
        (m.start(), m.end())
    });
    assert_eq!(None, cap.get(2));
    assert_eq!((2, 3), {
        let m = cap.get(4).unwrap();
        (m.start(), m.end())
    });

    assert_eq!(t!("abc"), match_text!(cap.get(0).unwrap()));
    assert_eq!(None, cap.get(2));
    assert_eq!(t!("c"), match_text!(cap.get(4).unwrap()));

    assert_eq!(None, cap.name("a"));
    assert_eq!(t!("c"), match_text!(cap.name("b").unwrap()));
}

#[test]
fn sub_capture_matches() {
    let re = regex!(r"([a-z])(([a-z])|([0-9]))");
    let cap = re.captures(t!("a5")).unwrap();
    let subs: Vec<_> = cap.iter().collect();

    assert_eq!(5, subs.len());
    assert!(subs[0].is_some());
    assert!(subs[1].is_some());
    assert!(subs[2].is_some());
    assert!(subs[3].is_none());
    assert!(subs[4].is_some());

    assert_eq!(t!("a5"), match_text!(subs[0].unwrap()));
    assert_eq!(t!("a"), match_text!(subs[1].unwrap()));
    assert_eq!(t!("5"), match_text!(subs[2].unwrap()));
    assert_eq!(t!("5"), match_text!(subs[4].unwrap()));
}

expand!(expand1, r"(?-u)(?P<foo>\w+)", "abc", "$foo", "abc");
expand!(expand2, r"(?-u)(?P<foo>\w+)", "abc", "$0", "abc");
expand!(expand3, r"(?-u)(?P<foo>\w+)", "abc", "$1", "abc");
expand!(expand4, r"(?-u)(?P<foo>\w+)", "abc", "$$1", "$1");
expand!(expand5, r"(?-u)(?P<foo>\w+)", "abc", "$$foo", "$foo");
expand!(expand6, r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)", "abc 123", "$b$a", "123abc");
expand!(expand7, r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)", "abc 123", "z$bz$az", "z");
expand!(
    expand8,
    r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)",
    "abc 123",
    ".$b.$a.",
    ".123.abc."
);
expand!(
    expand9,
    r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)",
    "abc 123",
    " $b $a ",
    " 123 abc "
);
expand!(expand10, r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)", "abc 123", "$bz$az", "");

expand!(expand_name1, r"%(?P<Z>[a-z]+)", "%abc", "$Z%", "abc%");
expand!(expand_name2, r"\[(?P<Z>[a-z]+)", "[abc", "$Z[", "abc[");
expand!(expand_name3, r"\{(?P<Z>[a-z]+)", "{abc", "$Z{", "abc{");
expand!(expand_name4, r"\}(?P<Z>[a-z]+)", "}abc", "$Z}", "abc}");
expand!(expand_name5, r"%([a-z]+)", "%abc", "$1a%", "%");
expand!(expand_name6, r"%([a-z]+)", "%abc", "${1}a%", "abca%");
expand!(expand_name7, r"\[(?P<Z[>[a-z]+)", "[abc", "${Z[}[", "abc[");
expand!(expand_name8, r"\[(?P<Z[>[a-z]+)", "[abc", "${foo}[", "[");
expand!(expand_name9, r"\[(?P<Z[>[a-z]+)", "[abc", "${1a}[", "[");
expand!(expand_name10, r"\[(?P<Z[>[a-z]+)", "[abc", "${#}[", "[");
expand!(expand_name11, r"\[(?P<Z[>[a-z]+)", "[abc", "${$$}[", "[");

split!(
    split1,
    r"(?-u)\s+",
    "a b\nc\td\n\t e",
    &[t!("a"), t!("b"), t!("c"), t!("d"), t!("e")]
);
split!(
    split2,
    r"(?-u)\b",
    "a b c",
    &[t!(""), t!("a"), t!(" "), t!("b"), t!(" "), t!("c"), t!("")]
);
split!(split3, r"a$", "a", &[t!(""), t!("")]);
split!(split_none, r"-", r"a", &[t!("a")]);
split!(split_trailing_blank, r"-", r"a-", &[t!("a"), t!("")]);
split!(split_trailing_blanks, r"-", r"a--", &[t!("a"), t!(""), t!("")]);
split!(split_empty, r"-", r"", &[t!("")]);

splitn!(splitn_below_limit, r"-", r"a", 2, &[t!("a")]);
splitn!(splitn_at_limit, r"-", r"a-b", 2, &[t!("a"), t!("b")]);
splitn!(splitn_above_limit, r"-", r"a-b-c", 2, &[t!("a"), t!("b-c")]);
splitn!(splitn_zero_limit, r"-", r"a-b", 0, empty_vec!());
splitn!(splitn_trailing_blank, r"-", r"a-", 2, &[t!("a"), t!("")]);
splitn!(splitn_trailing_separator, r"-", r"a--", 2, &[t!("a"), t!("-")]);
splitn!(splitn_empty, r"-", r"", 1, &[t!("")]);
matset!(set1, &["a", "a"], "a", 0, 1);
matset!(set2, &["a", "a"], "ba", 0, 1);
matset!(set3, &["a", "b"], "a", 0);
matset!(set4, &["a", "b"], "b", 1);
matset!(set5, &["a|b", "b|a"], "b", 0, 1);
matset!(set6, &["foo", "oo"], "foo", 0, 1);
matset!(set7, &["^foo", "bar$"], "foo", 0);
matset!(set8, &["^foo", "bar$"], "foo bar", 0, 1);
matset!(set9, &["^foo", "bar$"], "bar", 1);
matset!(set10, &[r"[a-z]+$", "foo"], "01234 foo", 0, 1);
matset!(set11, &[r"[a-z]+$", "foo"], "foo 01234", 1);
matset!(set12, &[r".*?", "a"], "zzzzzza", 0, 1);
matset!(set13, &[r".*", "a"], "zzzzzza", 0, 1);
matset!(set14, &[r".*", "a"], "zzzzzz", 0);
matset!(set15, &[r"(?-u)\ba\b"], "hello a bye", 0);
matset!(set16, &["a"], "a", 0);
matset!(set17, &[".*a"], "a", 0);
matset!(set18, &["a", "β"], "β", 1);

// regexes that match the empty string
matset!(setempty1, &["", "a"], "abc", 0, 1);
matset!(setempty2, &["", "b"], "abc", 0, 1);
matset!(setempty3, &["", "z"], "abc", 0);
matset!(setempty4, &["a", ""], "abc", 0, 1);
matset!(setempty5, &["b", ""], "abc", 0, 1);
matset!(setempty6, &["z", ""], "abc", 1);
matset!(setempty7, &["b", "(?:)"], "abc", 0, 1);
matset!(setempty8, &["(?:)", "b"], "abc", 0, 1);
matset!(setempty9, &["c(?:)", "b"], "abc", 0, 1);

nomatset!(nset1, &["a", "a"], "b");
nomatset!(nset2, &["^foo", "bar$"], "bar foo");
nomatset!(
    nset3,
    {
        let xs: &[&str] = &[];
        xs
    },
    "a"
);
nomatset!(nset4, &[r"^rooted$", r"\.log$"], "notrooted");

// See: https://github.com/rust-lang/regex/issues/187
#[test]
fn regression_subsequent_matches() {
    let set = regex_set!(&["ab", "b"]);
    let text = text!("ba");
    assert!(set.matches(text).matched(1));
    assert!(set.matches(text).matched(1));
}

#[test]
fn get_set_patterns() {
    let set = regex_set!(&["a", "b"]);
    assert_eq!(vec!["a", "b"], set.patterns());
}

#[test]
fn len_and_empty() {
    let empty = regex_set!(&[""; 0]);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let not_empty = regex_set!(&["ab", "b"]);
    assert_eq!(not_empty.len(), 2);
    assert!(!not_empty.is_empty());
}
//! On-board user LEDs
//!
//! - Red = Pin 22
//! - Green = Pin 19
//! - Blue = Pin 21
use embedded_hal::digital::v2::OutputPin;
use e310x_hal::gpio::gpio0::{Pin19, Pin21, Pin22};
use e310x_hal::gpio::{Output, Regular, Invert};

/// Red LED
pub type RED = Pin22<Output<Regular<Invert>>>;

/// Green LED
pub type GREEN = Pin19<Output<Regular<Invert>>>;

/// Blue LED
pub type BLUE = Pin21<Output<Regular<Invert>>>;

/// Returns RED, GREEN and BLUE LEDs.
pub fn rgb<X, Y, Z>(
    red: Pin22<X>, green: Pin19<Y>, blue: Pin21<Z>
) -> (RED, GREEN, BLUE)
{
    let red: RED = red.into_inverted_output();
    let green: GREEN = green.into_inverted_output();
    let blue: BLUE = blue.into_inverted_output();
    (red, green, blue)
}

/// Generic LED
pub trait Led {
    /// Turns the LED off
    fn off(&mut self);

    /// Turns the LED on
    fn on(&mut self);
}

impl Led for RED {
    fn off(&mut self) {
        self.set_low().unwrap();
    }

    fn on(&mut self) {
        self.set_high().unwrap();
    }
}

impl Led for GREEN {
    fn off(&mut self) {
        self.set_low().unwrap();
    }

    fn on(&mut self) {
        self.set_high().unwrap();
    }
}

impl Led for BLUE {
    fn off(&mut self) {
        self.set_low().unwrap();
    }

    fn on(&mut self) {
        self.set_high().unwrap();
    }
}
//! Board support crate for HiFive1 and LoFive boards

#![deny(missing_docs)]
#![no_std]

pub use e310x_hal as hal;

pub mod clock;
pub use clock::configure as configure_clocks;

pub mod flash;

#[cfg(any(feature = "board-hifive1", feature = "board-hifive1-revb"))]
pub mod led;
#[cfg(any(feature = "board-hifive1", feature = "board-hifive1-revb"))]
pub use led::{RED, GREEN, BLUE, rgb, Led};

pub mod stdout;
pub use stdout::configure as configure_stdout;

#[doc(hidden)]
pub mod gpio;
use std::{collections::BTreeMap, fmt::format, fs};

use deltae::{DEMethod::DE2000, DeltaE, LabValue};
use image::{
    imageops::{
        overlay,
        FilterType::{Gaussian, Nearest},
    },
    io::Reader,
    DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgba, RgbaImage,
};
use lab::Lab;
use silicon::{
    formatter::{ImageFormatter, ImageFormatterBuilder},
    utils::init_syntect,
};
use syntect::{
    easy::HighlightLines,
    highlighting::{Theme, ThemeSet},
    parsing::SyntaxSet,
    util::LinesWithEndings,
};
use tree_sitter::{Parser, Query, QueryCapture, QueryCursor};

#[derive(Copy, Clone, Debug)]
enum Language {
    Rust,
    C,
}

const SUB_SIZE: u32 = 400;
const TILE_NUM: u32 = 50;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let original_image = Reader::open("me.jpg")?.decode()?;
    let small = original_image.resize(TILE_NUM, TILE_NUM, image::imageops::FilterType::Gaussian);
    small.save("test.jpg").unwrap();
    let (ps, ts) = init_syntect();
    let funcs_rust = get_funcs(Language::Rust, "rust.rs");
    let funcs_c = get_funcs(Language::C, "c.c");
    //funcs.append(&mut funcs_2);
    /*let mut best_pics = vec![];
    let mut best_pics_handles = vec![];

    for pixel in small.pixels() {
        let color = pixel.2;
        let mut lang = Language::Rust;
        let mut codes = &mut funcs_rust;
        if codes.is_empty() {
            lang = Language::C;
            codes = &mut funcs_c;
        }
        let code = codes.pop().unwrap();
        let h = tokio::spawn(calculate_single(code, lang, color.clone(), ts.themes.clone(), ps.clone()));
        best_pics_handles.push(h);
    }
    println!("Done spawning");
    for h in best_pics_handles{
        best_pics.push(h.await);
    }*/
    let mut label_funcs_rust: Vec<(String, Language)> = funcs_rust
        .into_iter()
        .map(|s| (s, Language::Rust))
        .collect();
    let mut label_funcs_c: Vec<(String, Language)> =
        funcs_c.into_iter().map(|s| (s, Language::C)).collect();
    label_funcs_rust.append(&mut label_funcs_c);
    let mut funcs = label_funcs_rust;
    funcs.reverse();
    let mut row_handles = vec![];
    let width = small.width();
    for x in 0..small.height() {
        let mut code_vec = funcs.split_off(funcs.len() - width as usize);
        let pixel_row: Vec<(u32, u32, Rgba<u8>)> = small
            .pixels()
            .collect::<Vec<(u32, u32, Rgba<u8>)>>()
            .to_vec();
        let mut pixel_row = pixel_row[(x * width) as usize..((x + 1) * width) as usize].to_vec();
        //.to_vec()[(x*width) as usize..((x+1)*width) as usize];
        let themes = ts.themes.clone();
        let ps_t = ps.clone();
        let h = tokio::spawn(async move {
            let mut best_pics = vec![];
            for y in 0..width {
                let color = pixel_row.pop().unwrap().2;
                let (code, lang) = code_vec.pop().unwrap();
                let res = calculate_single(code, lang, color, themes.clone(), ps_t.clone()).await;

                best_pics.push(res.resize(SUB_SIZE, SUB_SIZE, Nearest));
                //best_pics.push(res);
            }
            best_pics
        });
        row_handles.push(h);
    }
    println!("Handles made!");
    //let mut pics = vec![];
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
        canvas.save(format!("progress/{}.jpeg", y_cntr)).unwrap();
        //pics.append(&mut h.await.unwrap());
        y_cntr += 1;
    }
    //for f in funcs {
    //    produce_image(&f, Language::Rust, &mut formatter, theme, &ps);
    //}
    Ok(())
}
fn new_formatter() -> ImageFormatter {
    ImageFormatterBuilder::new()
        .line_pad(5)
        .window_controls(false)
        .line_number(true)
        .round_corner(false)
        .tab_width(1)
        .font(vec![("Source Code Pro", 10.0), ("Fira Code", 10.0)])
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
        let new_dist = get_distance(&code_image, &color);
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

fn get_distance(img: &DynamicImage, color_b: &Rgba<u8>) -> f32 {
    let one = img.resize_exact(1, 1, Nearest);
    let color_a = GenericImageView::get_pixel(&one, 0, 0);
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

fn read_env_var(var_name: &str) -> String {
    let err = format!("Missing environment variable: {}", var_name);
    std::env::var(var_name).expect(&err)
}
