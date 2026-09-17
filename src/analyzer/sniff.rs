//! a name is a claim, not evidence; kind comes from the leading bytes.

use super::Kind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sniffed {
    pub kind: Option<Kind>,
    pub media_type: &'static str,
    pub label: &'static str,
}

impl Sniffed {
    const fn known(kind: Kind, media_type: &'static str, label: &'static str) -> Self {
        Sniffed { kind: Some(kind), media_type, label }
    }

    const fn foreign(media_type: &'static str, label: &'static str) -> Self {
        Sniffed { kind: None, media_type, label }
    }
}

const UNKNOWN: Sniffed = Sniffed::foreign("application/octet-stream", "an unrecognised binary file");

// some pdfs carry junk before %PDF, every reader tolerates it
const HEADER_SLACK: usize = 1024;

pub fn sniff(bytes: &[u8]) -> Sniffed {
    if bytes.is_empty() {
        return Sniffed::foreign("application/octet-stream", "an empty file");
    }

    if let Some(found) = magic(bytes) {
        return found;
    }
    if find(&bytes[..bytes.len().min(HEADER_SLACK)], b"%PDF-").is_some() {
        return Sniffed::known(Kind::Pdf, "application/pdf", "a PDF");
    }
    if let Some(encoding) = text_bom(bytes) {
        return Sniffed::known(Kind::Text, encoding, "a text file");
    }
    if looks_like_svg(bytes) {
        return Sniffed::known(Kind::Image, "image/svg+xml", "an SVG image");
    }
    if looks_like_text(bytes) {
        return Sniffed::known(Kind::Text, "text/plain", "a text file");
    }
    UNKNOWN
}

fn magic(b: &[u8]) -> Option<Sniffed> {
    let starts = |sig: &[u8]| b.starts_with(sig);

    if starts(b"%PDF-") {
        return Some(Sniffed::known(Kind::Pdf, "application/pdf", "a PDF"));
    }

    if starts(b"\x89PNG\r\n\x1a\n") {
        return Some(Sniffed::known(Kind::Image, "image/png", "a PNG image"));
    }
    if starts(b"\xff\xd8\xff") {
        return Some(Sniffed::known(Kind::Image, "image/jpeg", "a JPEG image"));
    }
    if starts(b"GIF87a") || starts(b"GIF89a") {
        return Some(Sniffed::known(Kind::Image, "image/gif", "a GIF image"));
    }
    if starts(b"BM") && b.len() > 14 {
        return Some(Sniffed::known(Kind::Image, "image/bmp", "a BMP image"));
    }
    if starts(b"II\x2a\x00") || starts(b"MM\x00\x2a") {
        return Some(Sniffed::known(Kind::Image, "image/tiff", "a TIFF image"));
    }
    if starts(b"\x00\x00\x01\x00") && b.len() > 6 {
        return Some(Sniffed::known(Kind::Image, "image/x-icon", "an icon file"));
    }

    // form type at byte 8 says which
    if starts(b"RIFF") && b.len() >= 12 {
        return Some(match &b[8..12] {
            b"WEBP" => Sniffed::known(Kind::Image, "image/webp", "a WebP image"),
            b"WAVE" => Sniffed::known(Kind::Audio, "audio/wav", "a WAV audio file"),
            b"AVI " => Sniffed::known(Kind::Video, "video/x-msvideo", "an AVI video"),
            _ => Sniffed::foreign("application/octet-stream", "a RIFF container"),
        });
    }

    // ftyp at byte 4, brand right after
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        let brand = &b[8..12];
        return Some(match brand {
            b"avif" | b"avis" => Sniffed::known(Kind::Image, "image/avif", "an AVIF image"),
            b"heic" | b"heix" | b"heim" | b"heis" | b"hevc" | b"mif1" | b"msf1" => {
                Sniffed::known(Kind::Image, "image/heic", "a HEIC image")
            }
            b"M4A " | b"M4B " => Sniffed::known(Kind::Audio, "audio/mp4", "an M4A audio file"),
            b"qt  " => Sniffed::known(Kind::Video, "video/quicktime", "a QuickTime video"),
            _ => Sniffed::known(Kind::Video, "video/mp4", "an MP4 video"),
        });
    }

    if starts(b"ID3") || (b.len() >= 2 && b[0] == 0xff && (b[1] & 0xe6) == 0xe2) {
        return Some(Sniffed::known(Kind::Audio, "audio/mpeg", "an MP3 audio file"));
    }
    if starts(b"fLaC") {
        return Some(Sniffed::known(Kind::Audio, "audio/flac", "a FLAC audio file"));
    }
    if starts(b"OggS") {
        // opus, vorbis and theora all ride in ogg; codec name is in the first page
        let head = &b[..b.len().min(512)];
        if find(head, b"OpusHead").is_some() {
            return Some(Sniffed::known(Kind::Audio, "audio/ogg", "an Opus audio file"));
        }
        if find(head, b"theora").is_some() {
            return Some(Sniffed::known(Kind::Video, "video/ogg", "an Ogg video"));
        }
        return Some(Sniffed::known(Kind::Audio, "audio/ogg", "an Ogg audio file"));
    }
    if starts(b"MThd") {
        return Some(Sniffed::known(Kind::Audio, "audio/midi", "a MIDI file"));
    }
    if starts(b"\xff\xf1") || starts(b"\xff\xf9") {
        return Some(Sniffed::known(Kind::Audio, "audio/aac", "an AAC audio file"));
    }
    if starts(b"\x1a\x45\xdf\xa3") {
        // matroska and webm share a header, doctype follows
        let head = &b[..b.len().min(256)];
        if find(head, b"webm").is_some() {
            return Some(Sniffed::known(Kind::Video, "video/webm", "a WebM video"));
        }
        return Some(Sniffed::known(Kind::Video, "video/x-matroska", "a Matroska video"));
    }

    if starts(b"PK\x03\x04") || starts(b"PK\x05\x06") {
        let head = &b[..b.len().min(4096)];
        if find(head, b"word/").is_some() {
            return Some(Sniffed::foreign("application/vnd.openxmlformats", "a Word document"));
        }
        if find(head, b"ppt/").is_some() {
            return Some(Sniffed::foreign("application/vnd.openxmlformats", "a PowerPoint file"));
        }
        if find(head, b"xl/").is_some() {
            return Some(Sniffed::foreign("application/vnd.openxmlformats", "an Excel workbook"));
        }
        return Some(Sniffed::foreign("application/zip", "a ZIP archive"));
    }
    if starts(b"\x1f\x8b") {
        return Some(Sniffed::foreign("application/gzip", "a gzip archive"));
    }
    if starts(b"Rar!\x1a\x07") {
        return Some(Sniffed::foreign("application/vnd.rar", "a RAR archive"));
    }
    if starts(b"7z\xbc\xaf\x27\x1c") {
        return Some(Sniffed::foreign("application/x-7z-compressed", "a 7-Zip archive"));
    }
    if starts(b"\x7fELF") {
        return Some(Sniffed::foreign("application/x-executable", "a Linux executable"));
    }
    if starts(b"MZ") {
        return Some(Sniffed::foreign("application/vnd.microsoft.portable-executable",
                                     "a Windows executable"));
    }
    if starts(b"SQLite format 3\x00") {
        return Some(Sniffed::foreign("application/vnd.sqlite3", "a SQLite database"));
    }
    if starts(b"{\\rtf") {
        return Some(Sniffed::foreign("application/rtf", "an RTF document"));
    }
    if starts(b"\xd0\xcf\x11\xe0") {
        return Some(Sniffed::foreign("application/x-ole-storage",
                                     "an old Office document (.doc, .xls or .ppt)"));
    }
    None
}

pub fn text_bom(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\xef\xbb\xbf") {
        return Some("text/plain");
    }
    if bytes.starts_with(b"\xff\xfe") {
        return Some("text/plain; charset=utf-16le");
    }
    if bytes.starts_with(b"\xfe\xff") {
        return Some("text/plain; charset=utf-16be");
    }
    None
}

// binary residue is NULs and stray control codes, not a file that fails utf-8
pub fn looks_like_text(bytes: &[u8]) -> bool {
    let sample = &bytes[..bytes.len().min(8192)];
    if sample.is_empty() || sample.contains(&0) {
        return false;
    }

    // a cut multi-byte char at the tail is not evidence
    let decoded = match std::str::from_utf8(sample) {
        Ok(text) => Some(text),
        Err(e) if e.valid_up_to() + 4 >= sample.len() => {
            std::str::from_utf8(&sample[..e.valid_up_to()]).ok()
        }
        Err(_) => None,
    };

    if let Some(text) = decoded {
        if text.is_empty() {
            return false;
        }
        let odd = text
            .chars()
            .filter(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t' | '\u{c}'))
            .count();
        return odd * 100 < text.chars().count();
    }

    // fall back to a single byte encoding: ascii, whitespace and the high half
    let plausible = sample
        .iter()
        .filter(|b| matches!(b, 0x20..=0x7e | b'\n' | b'\r' | b'\t' | 0x0c | 0xa0..=0xff))
        .count();
    plausible * 100 >= sample.len() * 95
}

fn looks_like_svg(bytes: &[u8]) -> bool {
    if !looks_like_text(bytes) {
        return false;
    }
    let head = &bytes[..bytes.len().min(2048)];
    find(head, b"<svg").is_some() || find(head, b"<SVG").is_some()
}

// the header-claimed size is what matters, a bomb is a small file claiming huge
pub fn image_dimensions(bytes: &[u8]) -> Option<(u64, u64)> {
    let be32 = |at: usize| -> Option<u64> {
        let s = bytes.get(at..at + 4)?;
        Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as u64)
    };
    let le32 = |at: usize| -> Option<u64> {
        let s = bytes.get(at..at + 4)?;
        Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as u64)
    };
    let le16 = |at: usize| -> Option<u64> {
        let s = bytes.get(at..at + 2)?;
        Some(u16::from_le_bytes([s[0], s[1]]) as u64)
    };

    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some((be32(16)?, be32(20)?));
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some((le16(6)?, le16(8)?));
    }
    if bytes.starts_with(b"BM") {
        // height is signed, negative for top-down bitmaps
        let w = le32(18)? as u32 as i32;
        let h = le32(22)? as u32 as i32;
        return Some((w.unsigned_abs() as u64, h.unsigned_abs() as u64));
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return match bytes.get(12..16)? {
            // lossless: 14 bits each, minus one
            b"VP8L" => {
                let s = bytes.get(21..25)?;
                let bits = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                Some((((bits & 0x3fff) + 1) as u64, (((bits >> 14) & 0x3fff) + 1) as u64))
            }
            b"VP8X" => {
                let s = bytes.get(24..30)?;
                let w = u32::from_le_bytes([s[0], s[1], s[2], 0]) as u64 + 1;
                let h = u32::from_le_bytes([s[3], s[4], s[5], 0]) as u64 + 1;
                Some((w, h))
            }
            b"VP8 " => {
                let s = bytes.get(26..30)?;
                let w = u16::from_le_bytes([s[0], s[1]]) as u64 & 0x3fff;
                let h = u16::from_le_bytes([s[2], s[3]]) as u64 & 0x3fff;
                Some((w, h))
            }
            _ => None,
        };
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return jpeg_dimensions(bytes);
    }
    None
}

fn jpeg_dimensions(bytes: &[u8]) -> Option<(u64, u64)> {
    let mut i = 2;
    while i + 3 < bytes.len() {
        if bytes[i] != 0xff {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        // padding and standalone markers carry no length field
        if marker == 0xff || (0xd0..=0xd9).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        // all frame headers except the arithmetic and hierarchical oddities
        let is_frame = (0xc0..=0xcf).contains(&marker)
            && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
        if is_frame {
            let h = u16::from_be_bytes([*bytes.get(i + 5)?, *bytes.get(i + 6)?]) as u64;
            let w = u16::from_be_bytes([*bytes.get(i + 7)?, *bytes.get(i + 8)?]) as u64;
            return Some((w, h));
        }
        if len < 2 {
            return None;
        }
        i += 2 + len;
    }
    None
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_mp3_renamed_to_pdf_is_still_an_mp3() {
        let mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00some audio payload";
        let found = sniff(mp3);
        assert_eq!(found.kind, Some(Kind::Audio));
        assert_eq!(found.media_type, "audio/mpeg");
    }

    #[test]
    fn a_pdf_renamed_to_mp3_is_still_a_pdf() {
        let pdf = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\n";
        assert_eq!(sniff(pdf).kind, Some(Kind::Pdf));
    }

    #[test]
    fn a_pdf_header_may_start_a_little_late() {
        let mut bytes = vec![b'\n'; 200];
        bytes.extend_from_slice(b"%PDF-1.4\n");
        assert_eq!(sniff(&bytes).kind, Some(Kind::Pdf));

        let mut buried = vec![b'\n'; 4000];
        buried.extend_from_slice(b"%PDF-1.4\n");
        assert_ne!(sniff(&buried).kind, Some(Kind::Pdf));
    }

    #[test]
    fn the_media_kinds_are_told_apart() {
        let cases: Vec<(&[u8], Kind, &str)> = vec![
            (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", Kind::Image, "image/png"),
            (b"\xff\xd8\xff\xe0\x00\x10JFIF", Kind::Image, "image/jpeg"),
            (b"GIF89a\x10\x00\x10\x00", Kind::Image, "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", Kind::Image, "image/webp"),
            (b"RIFF\x00\x00\x00\x00WAVEfmt ", Kind::Audio, "audio/wav"),
            (b"RIFF\x00\x00\x00\x00AVI LIST", Kind::Video, "video/x-msvideo"),
            (b"fLaC\x00\x00\x00\x22", Kind::Audio, "audio/flac"),
            (b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00OpusHead", Kind::Audio, "audio/ogg"),
            (b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00", Kind::Video, "video/mp4"),
            (b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00", Kind::Audio, "audio/mp4"),
            (b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00", Kind::Image, "image/avif"),
            (b"\x1a\x45\xdf\xa3\x01\x00\x00\x00webm", Kind::Video, "video/webm"),
        ];
        for (bytes, kind, media_type) in cases {
            let found = sniff(bytes);
            assert_eq!(found.kind, Some(kind), "{media_type}");
            assert_eq!(found.media_type, media_type);
        }
    }

    #[test]
    fn containers_are_named_but_refused() {
        let zip = b"PK\x03\x04\x14\x00\x00\x00\x08\x00word/document.xml";
        assert_eq!(sniff(zip).kind, None);
        assert_eq!(sniff(zip).label, "a Word document");
        assert_eq!(sniff(b"MZ\x90\x00\x03").label, "a Windows executable");
        assert_eq!(sniff(b"\x1f\x8b\x08\x00").label, "a gzip archive");
        assert_eq!(sniff(b"").label, "an empty file");
    }

    #[test]
    fn prose_is_text_and_binary_residue_is_not() {
        assert_eq!(sniff(b"# Notes\n\nSome prose about $x^2$.\n").kind, Some(Kind::Text));
        assert_eq!(sniff("hello \u{4e16}\u{754c}".as_bytes()).kind, Some(Kind::Text));
        assert_eq!(sniff(b"col a,col b\n1,2\n").kind, Some(Kind::Text));

        let binary: Vec<u8> = (0u8..=255).cycle().take(2000).collect();
        assert_eq!(sniff(&binary).kind, None);
        assert!(!looks_like_text(b"has a \x00 null"));
    }

    #[test]
    fn older_encodings_are_still_text() {
        let latin1 = b"R\xe9sum\xe9 of the caf\xe9 experiment, \xb1 0.5\n";
        assert_eq!(sniff(latin1).kind, Some(Kind::Text));

        let mut utf16 = b"\xff\xfe".to_vec();
        for c in "notes".encode_utf16() {
            utf16.extend_from_slice(&c.to_le_bytes());
        }
        let found = sniff(&utf16);
        assert_eq!(found.kind, Some(Kind::Text));
        assert_eq!(found.media_type, "text/plain; charset=utf-16le");
    }

    #[test]
    fn an_svg_stays_an_image() {
        let svg = b"<?xml version=\"1.0\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>";
        let found = sniff(svg);
        assert_eq!(found.kind, Some(Kind::Image));
        assert_eq!(found.media_type, "image/svg+xml");
    }

    #[test]
    fn image_headers_give_up_their_dimensions() {
        let mut png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR".to_vec();
        png.extend_from_slice(&40_000u32.to_be_bytes());
        png.extend_from_slice(&50_000u32.to_be_bytes());
        assert_eq!(image_dimensions(&png), Some((40_000, 50_000)));

        let gif = b"GIF89a\x40\x01\x20\x01".to_vec();
        assert_eq!(image_dimensions(&gif), Some((320, 288)));

        let mut jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            .to_vec();
        jpeg.extend_from_slice(b"\xff\xc0\x00\x11\x08");
        jpeg.extend_from_slice(&900u16.to_be_bytes());
        jpeg.extend_from_slice(&1600u16.to_be_bytes());
        assert_eq!(image_dimensions(&jpeg), Some((1600, 900)));

        assert_eq!(image_dimensions(b"not an image"), None);
    }
}
