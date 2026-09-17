//! pdf to png, and the text layer too, both through hayro. each page isolated so one bad page doesn't kill the render.

use std::panic::AssertUnwindSafe;

use hayro::hayro_interpret::font::Glyph;
use hayro::hayro_interpret::hayro_cmap::BfString;
use hayro::hayro_interpret::{
    BlendMode, ClipPath, Context, Device, GlyphDrawMode, Image, InterpreterCache,
    InterpreterSettings, Paint, PathDrawMode, SoftMask, TransformExt, interpret_page,
};
use hayro::hayro_syntax::page::Page;
use kurbo::{Affine, BezPath, Point, Rect};

// ~144 dpi: legible for figures, no huge pngs
const RENDER_SCALE: f32 = 2.0;
const MAX_PNG_BYTES: usize = 8 * 1024 * 1024;

pub struct PagePng {
    // zero-based; pages can be skipped, headers must still line up
    pub index: usize,
    pub png: Vec<u8>,
}

pub fn page_count(bytes: &[u8]) -> usize {
    match hayro::hayro_syntax::Pdf::new(bytes.to_vec()) {
        Ok(pdf) => pdf.pages().len(),
        Err(_) => 0,
    }
}

// file is reparsed per batch; failed pages are skipped, so the result can be shorter than asked
pub fn render_pdf_range(bytes: &[u8], start: usize, count: usize) -> Vec<PagePng> {
    let Ok(pdf) = hayro::hayro_syntax::Pdf::new(bytes.to_vec()) else {
        return Vec::new();
    };
    let cache = hayro::RenderCache::new();
    let interp = hayro::hayro_interpret::InterpreterSettings::default();
    let mut pages = Vec::new();
    for (index, page) in pdf.pages().iter().enumerate().skip(start).take(count) {
        let rendered = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let pix = hayro::render(
                page,
                &cache,
                &interp,
                &hayro::RenderSettings {
                    x_scale: RENDER_SCALE,
                    y_scale: RENDER_SCALE,
                    bg_color: hayro::vello_cpu::color::palette::css::WHITE,
                    ..Default::default()
                },
            );
            encode_png(pix.width() as u32, pix.height() as u32, pix.data_as_u8_slice())
        }));
        if let Ok(Ok(png)) = rendered {
            pages.push(PagePng { index, png });
        }
    }
    pages
}

fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || width > 8000 || height > 8000 {
        return Err("page has absurd dimensions".into());
    }
    if rgba.len() != width as usize * height as usize * 4 {
        return Err("pixmap size mismatch".into());
    }
    // premultiplied, but the white background makes every alpha 255, so plain rgba
    let mut buf = Vec::new();
    let mut enc = png::Encoder::new(&mut buf, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .map_err(|e| e.to_string())?
        .write_image_data(rgba)
        .map_err(|e| e.to_string())?;
    if buf.len() > MAX_PNG_BYTES {
        return Err("page image too large".into());
    }
    Ok(buf)
}

// characters come from each glyph's font map; raw strings in word/latex pdfs are gibberish
pub fn extract_text(bytes: &[u8]) -> String {
    let Ok(pdf) = hayro::hayro_syntax::Pdf::new(bytes.to_vec()) else {
        return String::new();
    };
    let cache = InterpreterCache::new();
    let settings = InterpreterSettings::default();
    let mut pages = Vec::new();
    for (index, page) in pdf.pages().iter().enumerate() {
        let text = std::panic::catch_unwind(AssertUnwindSafe(|| page_text(page, &cache, &settings)))
            .unwrap_or_default();
        if !text.is_empty() {
            pages.push(format!("## p. {}\n\n{text}", index + 1));
        }
    }
    pages.join("\n\n")
}

fn page_text<'a>(
    page: &'a Page<'a>,
    cache: &InterpreterCache<'a>,
    settings: &InterpreterSettings,
) -> String {
    let (width, height) = page.render_dimensions();
    let mut ctx = Context::new(
        page.initial_transform(true).to_kurbo(),
        Rect::new(0.0, 0.0, width as f64, height as f64),
        cache,
        page.xref(),
        settings.clone(),
    );
    let mut device = TextDevice::default();
    interpret_page(page, &mut ctx, &mut device);
    device.text.trim().to_string()
}

struct Placed {
    text: String,
    origin: Point,
    end: Point,
    size: f64,
}

#[derive(Default)]
struct TextDevice {
    text: String,
    last: Option<Placed>,
}

impl TextDevice {
    // spacing is all geometry: pdfs rarely draw spaces, a lower baseline is a newline, a bigger gap a space
    fn place(&mut self, text: String, origin: Point, end: Point, size: f64) {
        if let Some(last) = &self.last {
            // fill-and-stroke draws each glyph twice
            if last.text == text && (last.origin - origin).hypot() < 0.01 {
                return;
            }
            let line = size.max(last.size).max(1.0);
            let drop = origin.y - last.origin.y;
            if drop.abs() > line * 0.5 {
                let trimmed = self.text.trim_end_matches(' ').len();
                self.text.truncate(trimmed);
                self.text.push_str(if drop > line * 1.8 || drop < 0.0 { "\n\n" } else { "\n" });
            } else if origin.x - last.end.x > line * 0.15 && !self.text.ends_with(char::is_whitespace) {
                self.text.push(' ');
            }
        }
        if text.trim().is_empty() {
            if !self.text.is_empty() && !self.text.ends_with(char::is_whitespace) {
                self.text.push(' ');
            }
        } else {
            self.text.push_str(&text);
        }
        self.last = Some(Placed { text, origin, end, size });
    }
}

// a ligature is its own codepoint, so the index sees "classi\ufb01cation" as a different word
fn unfold_ligatures(text: String) -> String {
    if !text.chars().any(|c| ('\u{FB00}'..='\u{FB06}').contains(&c)) {
        return text;
    }
    let mut out = String::with_capacity(text.len() + 2);
    for c in text.chars() {
        match c {
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{FB05}' | '\u{FB06}' => out.push_str("st"),
            c => out.push(c),
        }
    }
    out
}

impl<'a> Device<'a> for TextDevice {
    fn set_soft_mask(&mut self, _: Option<SoftMask<'a>>) {}
    fn set_blend_mode(&mut self, _: BlendMode) {}
    fn draw_path(&mut self, _: &BezPath, _: Affine, _: &Paint<'a>, _: &PathDrawMode) {}
    fn push_clip_path(&mut self, _: &ClipPath) {}
    fn push_transparency_group(&mut self, _: f32, _: Option<SoftMask<'a>>, _: BlendMode) {}
    fn draw_image(&mut self, _: Image<'a, '_>, _: Affine) {}
    fn pop_clip_path(&mut self) {}
    fn pop_transparency_group(&mut self) {}

    fn draw_glyph(
        &mut self,
        glyph: &Glyph<'a>,
        transform: Affine,
        glyph_transform: Affine,
        _: &Paint<'a>,
        _: &GlyphDrawMode,
    ) {
        let text = match glyph.as_unicode() {
            Some(BfString::Char(c)) => c.to_string(),
            Some(BfString::String(s)) => s,
            None => return,
        };
        if text.chars().all(char::is_control) {
            return;
        }
        let text = unfold_ligatures(text);
        // glyph space and advance widths are both 1000 units per em
        let placed = transform * glyph_transform;
        let origin = placed * Point::ORIGIN;
        let size = (placed * Point::new(0.0, 1000.0) - origin).hypot();
        let advance = match glyph {
            Glyph::Outline(g) => g.advance_width().map(f64::from).filter(|a| *a > 0.0),
            Glyph::Type3(_) => None,
        }
        .unwrap_or(500.0);
        let end = placed * Point::new(advance, 0.0);
        self.place(text, origin, end, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_pdf() -> Vec<u8> {
        let content = b"1 0 0 rg 0 0 100 100 re f\n";
        let objects: Vec<Vec<u8>> = vec![
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_vec(),
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_vec(),
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
               /Contents 4 0 R >>\nendobj\n"
                .to_vec(),
            [
                format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).into_bytes(),
                content.to_vec(),
                b"\nendstream\nendobj\n".to_vec(),
            ]
            .concat(),
        ];
        build_pdf(&objects)
    }

    fn text_pdf() -> Vec<u8> {
        let content = b"BT /F1 24 Tf 72 700 Td (Hello world) Tj 0 -30 Td (Second line) Tj ET\n";
        build_pdf(&[
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_vec(),
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_vec(),
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
               /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
                .to_vec(),
            [
                format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).into_bytes(),
                content.to_vec(),
                b"\nendstream\nendobj\n".to_vec(),
            ]
            .concat(),
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
               /Encoding /WinAnsiEncoding >>\nendobj\n"
                .to_vec(),
        ])
    }

    fn build_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for obj in objects {
            offsets.push(pdf.len());
            pdf.extend_from_slice(obj);
        }
        let xref_at = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF",
                objects.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }

    #[test]
    fn renders_a_page_to_png() {
        let pdf = tiny_pdf();
        assert_eq!(page_count(&pdf), 1);
        let pages = render_pdf_range(&pdf, 0, 4);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].index, 0);
        assert_eq!(&pages[0].png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(pages[0].png.len() > 1000);
    }

    #[test]
    fn a_range_past_the_end_is_simply_empty() {
        let pdf = tiny_pdf();
        assert!(render_pdf_range(&pdf, 4, 4).is_empty());
    }

    #[test]
    fn reads_text_through_its_fonts_with_lines_and_spaces() {
        assert_eq!(extract_text(&text_pdf()), "## p. 1\n\nHello world\nSecond line");
    }

    #[test]
    fn ligatures_come_back_as_letters() {
        assert_eq!(unfold_ligatures("classi\u{FB01}cation".into()), "classification");
        assert_eq!(unfold_ligatures("Un\u{FB02}attering".into()), "Unflattering");
        assert_eq!(unfold_ligatures("plain".into()), "plain");
    }

    #[test]
    fn a_page_with_no_text_gives_no_header() {
        assert_eq!(extract_text(&tiny_pdf()), "");
        assert_eq!(extract_text(b"this is not a pdf at all"), "");
    }

    #[test]
    fn garbage_input_yields_nothing_without_panicking() {
        assert_eq!(page_count(b"this is not a pdf at all"), 0);
        assert!(render_pdf_range(b"this is not a pdf at all", 0, 4).is_empty());
        assert_eq!(page_count(b""), 0);
        assert!(render_pdf_range(b"", 0, 4).is_empty());
    }
}
