/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package kotlin.text

//
// NOTE: THIS FILE IS AUTO-GENERATED by the GenerateUnicodeData.kt
// See: https://github.com/JetBrains/kotlin/tree/master/libraries/stdlib
//

private val rangeStart = intArrayOf(
    0x0000, 0x0020, 0x0022, 0x0024, 0x0026, 0x0028, 0x002a, 0x002d, 0x002f, 0x0031, 0x003a, 0x003c, 0x003f, 0x0041, 0x005b, 0x005d, 0x005f, 0x0061, 0x007b, 0x007d, 
    0x007f, 0x00a0, 0x00a2, 0x00a6, 0x00a8, 0x00aa, 0x00ac, 0x00ae, 0x00b1, 0x00b3, 0x00b5, 0x00b7, 0x00b9, 0x00bb, 0x00bd, 0x00bf, 0x00c1, 0x00d7, 0x00d9, 0x00df, 
    0x00f7, 0x00f9, 0x0100, 0x0138, 0x0149, 0x0179, 0x017f, 0x0181, 0x0183, 0x0187, 0x018a, 0x018c, 0x018e, 0x0192, 0x0194, 0x0197, 0x0199, 0x019c, 0x019e, 0x01a0, 
    0x01a7, 0x01ab, 0x01af, 0x01b2, 0x01b4, 0x01b8, 0x01ba, 0x01bc, 0x01be, 0x01c0, 0x01c4, 0x01c6, 0x01c8, 0x01ca, 0x01cc, 0x01dd, 0x01f0, 0x01f2, 0x01f4, 0x01f7, 
    0x01f9, 0x0234, 0x023a, 0x023c, 0x023e, 0x0240, 0x0244, 0x0247, 0x0250, 0x0294, 0x0296, 0x02b0, 0x02c2, 0x02c6, 0x02d2, 0x02e0, 0x02e5, 0x02ec, 0x02f0, 0x0300, 
    0x0370, 0x0374, 0x0376, 0x037a, 0x037c, 0x037e, 0x0384, 0x0386, 0x0389, 0x038c, 0x038e, 0x0390, 0x0392, 0x03a3, 0x03ac, 0x03cf, 0x03d1, 0x03d3, 0x03d5, 0x03d8, 
    0x03f0, 0x03f4, 0x03f6, 0x03f8, 0x03fa, 0x03fc, 0x03fe, 0x0430, 0x0460, 0x0482, 0x0484, 0x0488, 0x048a, 0x04c1, 0x04cf, 0x0531, 0x0559, 0x055b, 0x0560, 0x0589, 
    0x058d, 0x058f, 0x0591, 0x05be, 0x05c0, 0x05c2, 0x05c5, 0x05d0, 0x05ef, 0x05f3, 0x0600, 0x0606, 0x0609, 0x060b, 0x060d, 0x060f, 0x0611, 0x061b, 0x061e, 0x0620, 
    0x0640, 0x0642, 0x064b, 0x0660, 0x066a, 0x066e, 0x0670, 0x0672, 0x06d4, 0x06d6, 0x06dd, 0x06df, 0x06e5, 0x06e7, 0x06e9, 0x06eb, 0x06ee, 0x06f0, 0x06fa, 0x06fd, 
    0x06ff, 0x0701, 0x070f, 0x0711, 0x0713, 0x0730, 0x074d, 0x07a6, 0x07b1, 0x07c0, 0x07ca, 0x07eb, 0x07f4, 0x07f6, 0x07f8, 0x07fa, 0x07fd, 0x07ff, 0x0801, 0x0816, 
    0x081a, 0x081c, 0x0824, 0x0826, 0x0828, 0x082a, 0x0830, 0x0840, 0x0859, 0x085e, 0x0860, 0x08a0, 0x08b6, 0x08d3, 0x08e2, 0x08e4, 0x0903, 0x0905, 0x093a, 0x093d, 
    0x093f, 0x0941, 0x0949, 0x094d, 0x094f, 0x0951, 0x0958, 0x0962, 0x0964, 0x0966, 0x0970, 0x0972, 0x0981, 0x0983, 0x0985, 0x098f, 0x0993, 0x09aa, 0x09b2, 0x09b6, 
    0x09bc, 0x09be, 0x09c1, 0x09c7, 0x09cb, 0x09cd, 0x09d7, 0x09dc, 0x09df, 0x09e2, 0x09e6, 0x09f0, 0x09f2, 0x09f4, 0x09fa, 0x09fc, 0x09fe, 0x0a01, 0x0a03, 0x0a05, 
    0x0a0f, 0x0a13, 0x0a2a, 0x0a32, 0x0a35, 0x0a38, 0x0a3c, 0x0a3e, 0x0a41, 0x0a47, 0x0a4b, 0x0a51, 0x0a59, 0x0a5e, 0x0a66, 0x0a70, 0x0a72, 0x0a75, 0x0a81, 0x0a83, 
    0x0a85, 0x0a8f, 0x0a93, 0x0aaa, 0x0ab2, 0x0ab5, 0x0abc, 0x0abe, 0x0ac1, 0x0ac7, 0x0ac9, 0x0acb, 0x0acd, 0x0ad0, 0x0ae0, 0x0ae2, 0x0ae6, 0x0af0, 0x0af9, 0x0afb, 
    0x0b01, 0x0b03, 0x0b05, 0x0b0f, 0x0b13, 0x0b2a, 0x0b32, 0x0b35, 0x0b3c, 0x0b3e, 0x0b42, 0x0b47, 0x0b4b, 0x0b4d, 0x0b55, 0x0b57, 0x0b5c, 0x0b5f, 0x0b62, 0x0b66, 
    0x0b70, 0x0b72, 0x0b82, 0x0b85, 0x0b8e, 0x0b92, 0x0b99, 0x0b9c, 0x0b9e, 0x0ba3, 0x0ba8, 0x0bae, 0x0bbe, 0x0bc0, 0x0bc2, 0x0bc6, 0x0bca, 0x0bcd, 0x0bd0, 0x0bd7, 
    0x0be6, 0x0bf0, 0x0bf3, 0x0bf9, 0x0c00, 0x0c02, 0x0c04, 0x0c06, 0x0c0e, 0x0c12, 0x0c2a, 0x0c3d, 0x0c3f, 0x0c41, 0x0c46, 0x0c4a, 0x0c55, 0x0c58, 0x0c60, 0x0c62, 
    0x0c66, 0x0c77, 0x0c79, 0x0c7f, 0x0c81, 0x0c83, 0x0c85, 0x0c8e, 0x0c92, 0x0caa, 0x0cb5, 0x0cbc, 0x0cbe, 0x0cc1, 0x0cc6, 0x0cc8, 0x0cca, 0x0ccc, 0x0cd5, 0x0cde, 
    0x0ce0, 0x0ce2, 0x0ce6, 0x0cf1, 0x0d00, 0x0d02, 0x0d04, 0x0d0e, 0x0d12, 0x0d3b, 0x0d3d, 0x0d3f, 0x0d41, 0x0d46, 0x0d4a, 0x0d4d, 0x0d4f, 0x0d54, 0x0d57, 0x0d59, 
    0x0d5f, 0x0d62, 0x0d66, 0x0d70, 0x0d79, 0x0d7b, 0x0d81, 0x0d83, 0x0d85, 0x0d9a, 0x0db3, 0x0dbd, 0x0dc0, 0x0dca, 0x0dcf, 0x0dd2, 0x0dd6, 0x0dd8, 0x0de6, 0x0df2, 
    0x0df4, 0x0e01, 0x0e31, 0x0e33, 0x0e35, 0x0e3f, 0x0e41, 0x0e46, 0x0e48, 0x0e4f, 0x0e51, 0x0e5a, 0x0e81, 0x0e84, 0x0e86, 0x0e8c, 0x0ea5, 0x0ea7, 0x0eb1, 0x0eb3, 
    0x0eb5, 0x0ebd, 0x0ec0, 0x0ec6, 0x0ec8, 0x0ed0, 0x0edc, 0x0f00, 0x0f02, 0x0f04, 0x0f13, 0x0f16, 0x0f18, 0x0f1a, 0x0f20, 0x0f2a, 0x0f34, 0x0f3a, 0x0f3e, 0x0f40, 
    0x0f49, 0x0f71, 0x0f7f, 0x0f81, 0x0f85, 0x0f87, 0x0f89, 0x0f8d, 0x0f99, 0x0fbe, 0x0fc6, 0x0fc8, 0x0fce, 0x0fd0, 0x0fd5, 0x0fd9, 0x1000, 0x102b, 0x102d, 0x1031, 
    0x1033, 0x1038, 0x103a, 0x103c, 0x103e, 0x1040, 0x104a, 0x1050, 0x1056, 0x1058, 0x105a, 0x105e, 0x1061, 0x1063, 0x1065, 0x1067, 0x106e, 0x1071, 0x1075, 0x1082, 
    0x1084, 0x1086, 0x1088, 0x108d, 0x108f, 0x1091, 0x109a, 0x109d, 0x109f, 0x10a1, 0x10c7, 0x10cd, 0x10d0, 0x10fb, 0x10fd, 0x1100, 0x124a, 0x1250, 0x1258, 0x125a, 
    0x1260, 0x128a, 0x1290, 0x12b2, 0x12b8, 0x12c0, 0x12c2, 0x12c8, 0x12d8, 0x1312, 0x1318, 0x135d, 0x1360, 0x1369, 0x1380, 0x1390, 0x13a0, 0x13f8, 0x1400, 0x1402, 
    0x166d, 0x166f, 0x1680, 0x1682, 0x169b, 0x16a0, 0x16eb, 0x16ee, 0x16f1, 0x1700, 0x170e, 0x1712, 0x1720, 0x1732, 0x1735, 0x1740, 0x1752, 0x1760, 0x176e, 0x1772, 
    0x1780, 0x17b4, 0x17b6, 0x17b8, 0x17be, 0x17c6, 0x17c8, 0x17ca, 0x17d4, 0x17d7, 0x17d9, 0x17db, 0x17dd, 0x17e0, 0x17f0, 0x1800, 0x1806, 0x1808, 0x180b, 0x180e, 
    0x1810, 0x1820, 0x1843, 0x1845, 0x1880, 0x1885, 0x1887, 0x18a9, 0x18b0, 0x1900, 0x1920, 0x1923, 0x1927, 0x1929, 0x1930, 0x1932, 0x1934, 0x1939, 0x1940, 0x1944, 
    0x1946, 0x1950, 0x1970, 0x1980, 0x19b0, 0x19d0, 0x19da, 0x19de, 0x1a00, 0x1a17, 0x1a19, 0x1a1b, 0x1a1e, 0x1a20, 0x1a55, 0x1a59, 0x1a60, 0x1a64, 0x1a66, 0x1a6d, 
    0x1a73, 0x1a7f, 0x1a81, 0x1a90, 0x1aa0, 0x1aa7, 0x1aa9, 0x1ab0, 0x1abe, 0x1ac0, 0x1b00, 0x1b04, 0x1b06, 0x1b34, 0x1b37, 0x1b3b, 0x1b3e, 0x1b42, 0x1b44, 0x1b46, 
    0x1b50, 0x1b5a, 0x1b61, 0x1b6b, 0x1b74, 0x1b80, 0x1b82, 0x1b84, 0x1ba1, 0x1ba3, 0x1ba6, 0x1ba8, 0x1baa, 0x1bac, 0x1bae, 0x1bb0, 0x1bba, 0x1be6, 0x1be9, 0x1beb, 
    0x1bed, 0x1bf0, 0x1bf2, 0x1bfc, 0x1c00, 0x1c24, 0x1c2c, 0x1c34, 0x1c36, 0x1c3b, 0x1c40, 0x1c4d, 0x1c50, 0x1c5a, 0x1c78, 0x1c7e, 0x1c80, 0x1c90, 0x1cbd, 0x1cc0, 
    0x1cd0, 0x1cd3, 0x1cd5, 0x1ce1, 0x1ce3, 0x1ce9, 0x1ced, 0x1cef, 0x1cf4, 0x1cf6, 0x1cf8, 0x1cfa, 0x1d00, 0x1d2c, 0x1d6b, 0x1d78, 0x1d7a, 0x1d9b, 0x1dc0, 0x1dfb, 
    0x1e00, 0x1e96, 0x1e9e, 0x1f00, 0x1f08, 0x1f10, 0x1f18, 0x1f20, 0x1f28, 0x1f30, 0x1f38, 0x1f40, 0x1f48, 0x1f50, 0x1f59, 0x1f5b, 0x1f5d, 0x1f5f, 0x1f61, 0x1f68, 
    0x1f70, 0x1f80, 0x1f88, 0x1f90, 0x1f98, 0x1fa0, 0x1fa8, 0x1fb0, 0x1fb6, 0x1fb8, 0x1fbc, 0x1fbe, 0x1fc0, 0x1fc2, 0x1fc6, 0x1fc8, 0x1fcc, 0x1fce, 0x1fd0, 0x1fd6, 
    0x1fd8, 0x1fdd, 0x1fe0, 0x1fe8, 0x1fed, 0x1ff2, 0x1ff6, 0x1ff8, 0x1ffc, 0x1ffe, 0x2000, 0x200b, 0x2010, 0x2016, 0x2018, 0x201a, 0x201c, 0x201e, 0x2020, 0x2028, 
    0x202a, 0x202f, 0x2031, 0x2039, 0x203b, 0x203f, 0x2041, 0x2044, 0x2046, 0x2048, 0x2052, 0x2054, 0x2056, 0x205f, 0x2061, 0x2066, 0x2070, 0x2074, 0x207a, 0x207d, 
    0x207f, 0x2081, 0x208a, 0x208d, 0x2090, 0x20a0, 0x20d0, 0x20dd, 0x20e1, 0x20e3, 0x20e5, 0x2100, 0x2102, 0x2104, 0x2107, 0x2109, 0x210b, 0x210e, 0x2110, 0x2113, 
    0x2115, 0x2117, 0x2119, 0x211e, 0x2124, 0x212b, 0x212e, 0x2130, 0x2134, 0x2136, 0x2139, 0x213b, 0x213d, 0x213f, 0x2141, 0x2145, 0x2147, 0x214a, 0x214d, 0x2150, 
    0x2160, 0x2183, 0x2185, 0x2189, 0x218b, 0x2190, 0x2195, 0x219a, 0x219c, 0x21a0, 0x21a2, 0x21a5, 0x21a8, 0x21ae, 0x21b0, 0x21ce, 0x21d0, 0x21d2, 0x21d6, 0x21f4, 
    0x2300, 0x2308, 0x230c, 0x2320, 0x2322, 0x2329, 0x232b, 0x237c, 0x237e, 0x239b, 0x23b4, 0x23dc, 0x23e2, 0x2440, 0x2460, 0x249c, 0x24ea, 0x2500, 0x25b7, 0x25b9, 
    0x25c1, 0x25c3, 0x25f8, 0x2600, 0x266f, 0x2671, 0x2768, 0x2776, 0x2794, 0x27c0, 0x27c5, 0x27c7, 0x27e6, 0x27f0, 0x2800, 0x2900, 0x2983, 0x2999, 0x29d8, 0x29dc, 
    0x29fc, 0x29fe, 0x2b00, 0x2b30, 0x2b45, 0x2b47, 0x2b4d, 0x2b76, 0x2b97, 0x2c00, 0x2c30, 0x2c60, 0x2c63, 0x2c65, 0x2c67, 0x2c6e, 0x2c71, 0x2c74, 0x2c77, 0x2c7c, 
    0x2c7e, 0x2c81, 0x2ce4, 0x2ce6, 0x2ceb, 0x2cef, 0x2cf2, 0x2cf9, 0x2cfd, 0x2cff, 0x2d01, 0x2d27, 0x2d2d, 0x2d30, 0x2d6f, 0x2d7f, 0x2d81, 0x2da0, 0x2da8, 0x2db0, 
    0x2db8, 0x2dc0, 0x2dc8, 0x2dd0, 0x2dd8, 0x2de0, 0x2e00, 0x2e02, 0x2e06, 0x2e09, 0x2e0b, 0x2e0d, 0x2e0f, 0x2e17, 0x2e19, 0x2e1c, 0x2e1e, 0x2e20, 0x2e22, 0x2e2a, 
    0x2e2f, 0x2e31, 0x2e3a, 0x2e3c, 0x2e40, 0x2e42, 0x2e44, 0x2e50, 0x2e52, 0x2e80, 0x2e9b, 0x2f00, 0x2ff0, 0x3000, 0x3002, 0x3004, 0x3006, 0x3008, 0x3012, 0x3014, 
    0x301c, 0x301e, 0x3020, 0x3022, 0x302a, 0x302e, 0x3030, 0x3032, 0x3036, 0x3038, 0x303b, 0x303d, 0x303f, 0x3041, 0x3099, 0x309b, 0x309d, 0x309f, 0x30a2, 0x30fb, 
    0x30fd, 0x30ff, 0x3105, 0x3131, 0x3190, 0x3192, 0x3196, 0x31a0, 0x31c0, 0x31f0, 0x3200, 0x3220, 0x322a, 0x3248, 0x3250, 0x3252, 0x3260, 0x3280, 0x328a, 0x32b1, 
    0x32c0, 0x3400, 0x4dc0, 0x4e00, 0xa000, 0xa015, 0xa017, 0xa490, 0xa4d0, 0xa4f8, 0xa4fe, 0xa500, 0xa60c, 0xa60e, 0xa610, 0xa620, 0xa62a, 0xa640, 0xa66e, 0xa670, 
    0xa673, 0xa675, 0xa67e, 0xa680, 0xa69c, 0xa69e, 0xa6a0, 0xa6e6, 0xa6f0, 0xa6f2, 0xa700, 0xa717, 0xa720, 0xa722, 0xa730, 0xa732, 0xa770, 0xa772, 0xa779, 0xa77e, 
    0xa788, 0xa78a, 0xa78c, 0xa78f, 0xa791, 0xa794, 0xa796, 0xa7ab, 0xa7af, 0xa7b1, 0xa7b5, 0xa7c2, 0xa7c5, 0xa7c8, 0xa7f5, 0xa7f7, 0xa7f9, 0xa7fb, 0xa802, 0xa804, 
    0xa806, 0xa808, 0xa80b, 0xa80d, 0xa823, 0xa825, 0xa827, 0xa829, 0xa82c, 0xa830, 0xa836, 0xa838, 0xa840, 0xa874, 0xa880, 0xa882, 0xa8b4, 0xa8c4, 0xa8ce, 0xa8d0, 
    0xa8e0, 0xa8f2, 0xa8f8, 0xa8fb, 0xa8fe, 0xa900, 0xa90a, 0xa926, 0xa92e, 0xa930, 0xa947, 0xa952, 0xa95f, 0xa961, 0xa980, 0xa983, 0xa985, 0xa9b3, 0xa9b5, 0xa9b7, 
    0xa9ba, 0xa9bc, 0xa9be, 0xa9c1, 0xa9cf, 0xa9d1, 0xa9de, 0xa9e0, 0xa9e5, 0xa9e7, 0xa9f0, 0xa9fa, 0xaa00, 0xaa29, 0xaa2f, 0xaa31, 0xaa33, 0xaa35, 0xaa40, 0xaa43, 
    0xaa45, 0xaa4c, 0xaa50, 0xaa5c, 0xaa60, 0xaa70, 0xaa72, 0xaa77, 0xaa7a, 0xaa7c, 0xaa7e, 0xaab0, 0xaab3, 0xaab5, 0xaab7, 0xaab9, 0xaabe, 0xaac0, 0xaadb, 0xaadd, 
    0xaadf, 0xaae1, 0xaaeb, 0xaaed, 0xaaef, 0xaaf1, 0xaaf3, 0xaaf5, 0xab01, 0xab09, 0xab11, 0xab20, 0xab28, 0xab30, 0xab5b, 0xab5d, 0xab60, 0xab69, 0xab6b, 0xab70, 
    0xabc0, 0xabe3, 0xabe5, 0xabe7, 0xabea, 0xabed, 0xabf0, 0xac00, 0xd7b0, 0xd7cb, 0xd800, 0xdb80, 0xdc00, 0xe000, 0xf900, 0xfa70, 0xfb00, 0xfb13, 0xfb1d, 0xfb20, 
    0xfb29, 0xfb2b, 0xfb38, 0xfb3e, 0xfb40, 0xfb43, 0xfb46, 0xfbb2, 0xfbd3, 0xfd3e, 0xfd50, 0xfd92, 0xfdf0, 0xfdfc, 0xfe00, 0xfe10, 0xfe17, 0xfe19, 0xfe20, 0xfe30, 
    0xfe32, 0xfe34, 0xfe36, 0xfe45, 0xfe47, 0xfe49, 0xfe4d, 0xfe50, 0xfe54, 0xfe58, 0xfe5a, 0xfe5f, 0xfe62, 0xfe65, 0xfe68, 0xfe6b, 0xfe70, 0xfe76, 0xfeff, 0xff01, 
    0xff04, 0xff06, 0xff08, 0xff0a, 0xff0d, 0xff0f, 0xff11, 0xff1a, 0xff1c, 0xff1f, 0xff21, 0xff3b, 0xff3d, 0xff3f, 0xff41, 0xff5b, 0xff5d, 0xff5f, 0xff61, 0xff63, 
    0xff65, 0xff67, 0xff70, 0xff72, 0xff9e, 0xffa0, 0xffc2, 0xffca, 0xffd2, 0xffda, 0xffe0, 0xffe2, 0xffe4, 0xffe6, 0xffe8, 0xffea, 0xffed, 0xfff9, 0xfffc, 
)

private val rangeEnd = intArrayOf(
    0x001f, 0x0021, 0x0023, 0x0025, 0x0027, 0x0029, 0x002c, 0x002e, 0x0030, 0x0039, 0x003b, 0x003e, 0x0040, 0x005a, 0x005c, 0x005e, 0x0060, 0x007a, 0x007c, 0x007e, 
    0x009f, 0x00a1, 0x00a5, 0x00a7, 0x00a9, 0x00ab, 0x00ad, 0x00b0, 0x00b2, 0x00b4, 0x00b6, 0x00b8, 0x00ba, 0x00bc, 0x00be, 0x00c0, 0x00d6, 0x00d8, 0x00de, 0x00f6, 
    0x00f8, 0x00ff, 0x0137, 0x0148, 0x0178, 0x017e, 0x0180, 0x0182, 0x0186, 0x0189, 0x018b, 0x018d, 0x0191, 0x0193, 0x0196, 0x0198, 0x019b, 0x019d, 0x019f, 0x01a6, 
    0x01aa, 0x01ae, 0x01b1, 0x01b3, 0x01b7, 0x01b9, 0x01bb, 0x01bd, 0x01bf, 0x01c3, 0x01c5, 0x01c7, 0x01c9, 0x01cb, 0x01dc, 0x01ef, 0x01f1, 0x01f3, 0x01f6, 0x01f8, 
    0x0233, 0x0239, 0x023b, 0x023d, 0x023f, 0x0243, 0x0246, 0x024f, 0x0293, 0x0295, 0x02af, 0x02c1, 0x02c5, 0x02d1, 0x02df, 0x02e4, 0x02eb, 0x02ef, 0x02ff, 0x036f, 
    0x0373, 0x0375, 0x0377, 0x037b, 0x037d, 0x037f, 0x0385, 0x0388, 0x038a, 0x038c, 0x038f, 0x0391, 0x03a1, 0x03ab, 0x03ce, 0x03d0, 0x03d2, 0x03d4, 0x03d7, 0x03ef, 
    0x03f3, 0x03f5, 0x03f7, 0x03f9, 0x03fb, 0x03fd, 0x042f, 0x045f, 0x0481, 0x0483, 0x0487, 0x0489, 0x04c0, 0x04ce, 0x052f, 0x0556, 0x055a, 0x055f, 0x0588, 0x058a, 
    0x058e, 0x058f, 0x05bd, 0x05bf, 0x05c1, 0x05c4, 0x05c7, 0x05ea, 0x05f2, 0x05f4, 0x0605, 0x0608, 0x060a, 0x060c, 0x060e, 0x0610, 0x061a, 0x061c, 0x061f, 0x063f, 
    0x0641, 0x064a, 0x065f, 0x0669, 0x066d, 0x066f, 0x0671, 0x06d3, 0x06d5, 0x06dc, 0x06de, 0x06e4, 0x06e6, 0x06e8, 0x06ea, 0x06ed, 0x06ef, 0x06f9, 0x06fc, 0x06fe, 
    0x0700, 0x070d, 0x0710, 0x0712, 0x072f, 0x074a, 0x07a5, 0x07b0, 0x07b1, 0x07c9, 0x07ea, 0x07f3, 0x07f5, 0x07f7, 0x07f9, 0x07fa, 0x07fe, 0x0800, 0x0815, 0x0819, 
    0x081b, 0x0823, 0x0825, 0x0827, 0x0829, 0x082d, 0x083e, 0x0858, 0x085b, 0x085e, 0x086a, 0x08b4, 0x08c7, 0x08e1, 0x08e3, 0x0902, 0x0904, 0x0939, 0x093c, 0x093e, 
    0x0940, 0x0948, 0x094c, 0x094e, 0x0950, 0x0957, 0x0961, 0x0963, 0x0965, 0x096f, 0x0971, 0x0980, 0x0982, 0x0983, 0x098c, 0x0990, 0x09a8, 0x09b0, 0x09b2, 0x09b9, 
    0x09bd, 0x09c0, 0x09c4, 0x09c8, 0x09cc, 0x09ce, 0x09d7, 0x09dd, 0x09e1, 0x09e3, 0x09ef, 0x09f1, 0x09f3, 0x09f9, 0x09fb, 0x09fd, 0x09fe, 0x0a02, 0x0a03, 0x0a0a, 
    0x0a10, 0x0a28, 0x0a30, 0x0a33, 0x0a36, 0x0a39, 0x0a3c, 0x0a40, 0x0a42, 0x0a48, 0x0a4d, 0x0a51, 0x0a5c, 0x0a5e, 0x0a6f, 0x0a71, 0x0a74, 0x0a76, 0x0a82, 0x0a83, 
    0x0a8d, 0x0a91, 0x0aa8, 0x0ab0, 0x0ab3, 0x0ab9, 0x0abd, 0x0ac0, 0x0ac5, 0x0ac8, 0x0ac9, 0x0acc, 0x0acd, 0x0ad0, 0x0ae1, 0x0ae3, 0x0aef, 0x0af1, 0x0afa, 0x0aff, 
    0x0b02, 0x0b03, 0x0b0c, 0x0b10, 0x0b28, 0x0b30, 0x0b33, 0x0b39, 0x0b3d, 0x0b41, 0x0b44, 0x0b48, 0x0b4c, 0x0b4d, 0x0b56, 0x0b57, 0x0b5d, 0x0b61, 0x0b63, 0x0b6f, 
    0x0b71, 0x0b77, 0x0b83, 0x0b8a, 0x0b90, 0x0b95, 0x0b9a, 0x0b9c, 0x0b9f, 0x0ba4, 0x0baa, 0x0bb9, 0x0bbf, 0x0bc1, 0x0bc2, 0x0bc8, 0x0bcc, 0x0bcd, 0x0bd0, 0x0bd7, 
    0x0bef, 0x0bf2, 0x0bf8, 0x0bfa, 0x0c01, 0x0c03, 0x0c05, 0x0c0c, 0x0c10, 0x0c28, 0x0c39, 0x0c3e, 0x0c40, 0x0c44, 0x0c48, 0x0c4d, 0x0c56, 0x0c5a, 0x0c61, 0x0c63, 
    0x0c6f, 0x0c78, 0x0c7e, 0x0c80, 0x0c82, 0x0c84, 0x0c8c, 0x0c90, 0x0ca8, 0x0cb3, 0x0cb9, 0x0cbd, 0x0cc0, 0x0cc4, 0x0cc7, 0x0cc8, 0x0ccb, 0x0ccd, 0x0cd6, 0x0cde, 
    0x0ce1, 0x0ce3, 0x0cef, 0x0cf2, 0x0d01, 0x0d03, 0x0d0c, 0x0d10, 0x0d3a, 0x0d3c, 0x0d3e, 0x0d40, 0x0d44, 0x0d48, 0x0d4c, 0x0d4e, 0x0d4f, 0x0d56, 0x0d58, 0x0d5e, 
    0x0d61, 0x0d63, 0x0d6f, 0x0d78, 0x0d7a, 0x0d7f, 0x0d82, 0x0d83, 0x0d96, 0x0db1, 0x0dbb, 0x0dbd, 0x0dc6, 0x0dca, 0x0dd1, 0x0dd4, 0x0dd6, 0x0ddf, 0x0def, 0x0df3, 
    0x0df4, 0x0e30, 0x0e32, 0x0e34, 0x0e3a, 0x0e40, 0x0e45, 0x0e47, 0x0e4e, 0x0e50, 0x0e59, 0x0e5b, 0x0e82, 0x0e84, 0x0e8a, 0x0ea3, 0x0ea5, 0x0eb0, 0x0eb2, 0x0eb4, 
    0x0ebc, 0x0ebd, 0x0ec4, 0x0ec6, 0x0ecd, 0x0ed9, 0x0edf, 0x0f01, 0x0f03, 0x0f12, 0x0f15, 0x0f17, 0x0f19, 0x0f1f, 0x0f29, 0x0f33, 0x0f39, 0x0f3d, 0x0f3f, 0x0f47, 
    0x0f6c, 0x0f7e, 0x0f80, 0x0f84, 0x0f86, 0x0f88, 0x0f8c, 0x0f97, 0x0fbc, 0x0fc5, 0x0fc7, 0x0fcc, 0x0fcf, 0x0fd4, 0x0fd8, 0x0fda, 0x102a, 0x102c, 0x1030, 0x1032, 
    0x1037, 0x1039, 0x103b, 0x103d, 0x103f, 0x1049, 0x104f, 0x1055, 0x1057, 0x1059, 0x105d, 0x1060, 0x1062, 0x1064, 0x1066, 0x106d, 0x1070, 0x1074, 0x1081, 0x1083, 
    0x1085, 0x1087, 0x108c, 0x108e, 0x1090, 0x1099, 0x109c, 0x109e, 0x10a0, 0x10c5, 0x10c7, 0x10cd, 0x10fa, 0x10fc, 0x10ff, 0x1248, 0x124d, 0x1256, 0x1258, 0x125d, 
    0x1288, 0x128d, 0x12b0, 0x12b5, 0x12be, 0x12c0, 0x12c5, 0x12d6, 0x1310, 0x1315, 0x135a, 0x135f, 0x1368, 0x137c, 0x138f, 0x1399, 0x13f5, 0x13fd, 0x1401, 0x166c, 
    0x166e, 0x167f, 0x1681, 0x169a, 0x169c, 0x16ea, 0x16ed, 0x16f0, 0x16f8, 0x170c, 0x1711, 0x1714, 0x1731, 0x1734, 0x1736, 0x1751, 0x1753, 0x176c, 0x1770, 0x1773, 
    0x17b3, 0x17b5, 0x17b7, 0x17bd, 0x17c5, 0x17c7, 0x17c9, 0x17d3, 0x17d6, 0x17d8, 0x17da, 0x17dc, 0x17dd, 0x17e9, 0x17f9, 0x1805, 0x1807, 0x180a, 0x180d, 0x180e, 
    0x1819, 0x1842, 0x1844, 0x1878, 0x1884, 0x1886, 0x18a8, 0x18aa, 0x18f5, 0x191e, 0x1922, 0x1926, 0x1928, 0x192b, 0x1931, 0x1933, 0x1938, 0x193b, 0x1940, 0x1945, 
    0x194f, 0x196d, 0x1974, 0x19ab, 0x19c9, 0x19d9, 0x19da, 0x19ff, 0x1a16, 0x1a18, 0x1a1a, 0x1a1b, 0x1a1f, 0x1a54, 0x1a58, 0x1a5e, 0x1a63, 0x1a65, 0x1a6c, 0x1a72, 
    0x1a7c, 0x1a80, 0x1a89, 0x1a99, 0x1aa6, 0x1aa8, 0x1aad, 0x1abd, 0x1abf, 0x1ac0, 0x1b03, 0x1b05, 0x1b33, 0x1b36, 0x1b3a, 0x1b3d, 0x1b41, 0x1b43, 0x1b45, 0x1b4b, 
    0x1b59, 0x1b60, 0x1b6a, 0x1b73, 0x1b7c, 0x1b81, 0x1b83, 0x1ba0, 0x1ba2, 0x1ba5, 0x1ba7, 0x1ba9, 0x1bab, 0x1bad, 0x1baf, 0x1bb9, 0x1be5, 0x1be8, 0x1bea, 0x1bec, 
    0x1bef, 0x1bf1, 0x1bf3, 0x1bff, 0x1c23, 0x1c2b, 0x1c33, 0x1c35, 0x1c37, 0x1c3f, 0x1c49, 0x1c4f, 0x1c59, 0x1c77, 0x1c7d, 0x1c7f, 0x1c88, 0x1cba, 0x1cbf, 0x1cc7, 
    0x1cd2, 0x1cd4, 0x1ce0, 0x1ce2, 0x1ce8, 0x1cec, 0x1cee, 0x1cf3, 0x1cf5, 0x1cf7, 0x1cf9, 0x1cfa, 0x1d2b, 0x1d6a, 0x1d77, 0x1d79, 0x1d9a, 0x1dbf, 0x1df9, 0x1dff, 
    0x1e95, 0x1e9d, 0x1eff, 0x1f07, 0x1f0f, 0x1f15, 0x1f1d, 0x1f27, 0x1f2f, 0x1f37, 0x1f3f, 0x1f45, 0x1f4d, 0x1f57, 0x1f59, 0x1f5b, 0x1f5d, 0x1f60, 0x1f67, 0x1f6f, 
    0x1f7d, 0x1f87, 0x1f8f, 0x1f97, 0x1f9f, 0x1fa7, 0x1faf, 0x1fb4, 0x1fb7, 0x1fbb, 0x1fbd, 0x1fbf, 0x1fc1, 0x1fc4, 0x1fc7, 0x1fcb, 0x1fcd, 0x1fcf, 0x1fd3, 0x1fd7, 
    0x1fdb, 0x1fdf, 0x1fe7, 0x1fec, 0x1fef, 0x1ff4, 0x1ff7, 0x1ffb, 0x1ffd, 0x1ffe, 0x200a, 0x200f, 0x2015, 0x2017, 0x2019, 0x201b, 0x201d, 0x201f, 0x2027, 0x2029, 
    0x202e, 0x2030, 0x2038, 0x203a, 0x203e, 0x2040, 0x2043, 0x2045, 0x2047, 0x2051, 0x2053, 0x2055, 0x205e, 0x2060, 0x2064, 0x206f, 0x2071, 0x2079, 0x207c, 0x207e, 
    0x2080, 0x2089, 0x208c, 0x208e, 0x209c, 0x20bf, 0x20dc, 0x20e0, 0x20e2, 0x20e4, 0x20f0, 0x2101, 0x2103, 0x2106, 0x2108, 0x210a, 0x210d, 0x210f, 0x2112, 0x2114, 
    0x2116, 0x2118, 0x211d, 0x2123, 0x212a, 0x212d, 0x212f, 0x2133, 0x2135, 0x2138, 0x213a, 0x213c, 0x213e, 0x2140, 0x2144, 0x2146, 0x2149, 0x214c, 0x214f, 0x215f, 
    0x2182, 0x2184, 0x2188, 0x218a, 0x218b, 0x2194, 0x2199, 0x219b, 0x219f, 0x21a1, 0x21a4, 0x21a7, 0x21ad, 0x21af, 0x21cd, 0x21cf, 0x21d1, 0x21d5, 0x21f3, 0x22ff, 
    0x2307, 0x230b, 0x231f, 0x2321, 0x2328, 0x232a, 0x237b, 0x237d, 0x239a, 0x23b3, 0x23db, 0x23e1, 0x2426, 0x244a, 0x249b, 0x24e9, 0x24ff, 0x25b6, 0x25b8, 0x25c0, 
    0x25c2, 0x25f7, 0x25ff, 0x266e, 0x2670, 0x2767, 0x2775, 0x2793, 0x27bf, 0x27c4, 0x27c6, 0x27e5, 0x27ef, 0x27ff, 0x28ff, 0x2982, 0x2998, 0x29d7, 0x29db, 0x29fb, 
    0x29fd, 0x2aff, 0x2b2f, 0x2b44, 0x2b46, 0x2b4c, 0x2b73, 0x2b95, 0x2bff, 0x2c2e, 0x2c5e, 0x2c62, 0x2c64, 0x2c66, 0x2c6d, 0x2c70, 0x2c73, 0x2c76, 0x2c7b, 0x2c7d, 
    0x2c80, 0x2ce3, 0x2ce5, 0x2cea, 0x2cee, 0x2cf1, 0x2cf3, 0x2cfc, 0x2cfe, 0x2d00, 0x2d25, 0x2d27, 0x2d2d, 0x2d67, 0x2d70, 0x2d80, 0x2d96, 0x2da6, 0x2dae, 0x2db6, 
    0x2dbe, 0x2dc6, 0x2dce, 0x2dd6, 0x2dde, 0x2dff, 0x2e01, 0x2e05, 0x2e08, 0x2e0a, 0x2e0c, 0x2e0e, 0x2e16, 0x2e18, 0x2e1b, 0x2e1d, 0x2e1f, 0x2e21, 0x2e29, 0x2e2e, 
    0x2e30, 0x2e39, 0x2e3b, 0x2e3f, 0x2e41, 0x2e43, 0x2e4f, 0x2e51, 0x2e52, 0x2e99, 0x2ef3, 0x2fd5, 0x2ffb, 0x3001, 0x3003, 0x3005, 0x3007, 0x3011, 0x3013, 0x301b, 
    0x301d, 0x301f, 0x3021, 0x3029, 0x302d, 0x302f, 0x3031, 0x3035, 0x3037, 0x303a, 0x303c, 0x303e, 0x303f, 0x3096, 0x309a, 0x309c, 0x309e, 0x30a1, 0x30fa, 0x30fc, 
    0x30fe, 0x30ff, 0x312f, 0x318e, 0x3191, 0x3195, 0x319f, 0x31bf, 0x31e3, 0x31ff, 0x321e, 0x3229, 0x3247, 0x324f, 0x3251, 0x325f, 0x327f, 0x3289, 0x32b0, 0x32bf, 
    0x33ff, 0x4dbf, 0x4dff, 0x9ffc, 0xa014, 0xa016, 0xa48c, 0xa4c6, 0xa4f7, 0xa4fd, 0xa4ff, 0xa60b, 0xa60d, 0xa60f, 0xa61f, 0xa629, 0xa62b, 0xa66d, 0xa66f, 0xa672, 
    0xa674, 0xa67d, 0xa67f, 0xa69b, 0xa69d, 0xa69f, 0xa6e5, 0xa6ef, 0xa6f1, 0xa6f7, 0xa716, 0xa71f, 0xa721, 0xa72f, 0xa731, 0xa76f, 0xa771, 0xa778, 0xa77d, 0xa787, 
    0xa789, 0xa78b, 0xa78e, 0xa790, 0xa793, 0xa795, 0xa7aa, 0xa7ae, 0xa7b0, 0xa7b4, 0xa7bf, 0xa7c4, 0xa7c7, 0xa7ca, 0xa7f6, 0xa7f8, 0xa7fa, 0xa801, 0xa803, 0xa805, 
    0xa807, 0xa80a, 0xa80c, 0xa822, 0xa824, 0xa826, 0xa828, 0xa82b, 0xa82c, 0xa835, 0xa837, 0xa839, 0xa873, 0xa877, 0xa881, 0xa8b3, 0xa8c3, 0xa8c5, 0xa8cf, 0xa8d9, 
    0xa8f1, 0xa8f7, 0xa8fa, 0xa8fd, 0xa8ff, 0xa909, 0xa925, 0xa92d, 0xa92f, 0xa946, 0xa951, 0xa953, 0xa960, 0xa97c, 0xa982, 0xa984, 0xa9b2, 0xa9b4, 0xa9b6, 0xa9b9, 
    0xa9bb, 0xa9bd, 0xa9c0, 0xa9cd, 0xa9d0, 0xa9d9, 0xa9df, 0xa9e4, 0xa9e6, 0xa9ef, 0xa9f9, 0xa9fe, 0xaa28, 0xaa2e, 0xaa30, 0xaa32, 0xaa34, 0xaa36, 0xaa42, 0xaa44, 
    0xaa4b, 0xaa4d, 0xaa59, 0xaa5f, 0xaa6f, 0xaa71, 0xaa76, 0xaa79, 0xaa7b, 0xaa7d, 0xaaaf, 0xaab2, 0xaab4, 0xaab6, 0xaab8, 0xaabd, 0xaabf, 0xaac2, 0xaadc, 0xaade, 
    0xaae0, 0xaaea, 0xaaec, 0xaaee, 0xaaf0, 0xaaf2, 0xaaf4, 0xaaf6, 0xab06, 0xab0e, 0xab16, 0xab26, 0xab2e, 0xab5a, 0xab5c, 0xab5f, 0xab68, 0xab6a, 0xab6b, 0xabbf, 
    0xabe2, 0xabe4, 0xabe6, 0xabe9, 0xabec, 0xabed, 0xabf9, 0xd7a3, 0xd7c6, 0xd7fb, 0xdb7f, 0xdbff, 0xdfff, 0xf8ff, 0xfa6d, 0xfad9, 0xfb06, 0xfb17, 0xfb1f, 0xfb28, 
    0xfb2a, 0xfb36, 0xfb3c, 0xfb3e, 0xfb41, 0xfb44, 0xfbb1, 0xfbc1, 0xfd3d, 0xfd3f, 0xfd8f, 0xfdc7, 0xfdfb, 0xfdfd, 0xfe0f, 0xfe16, 0xfe18, 0xfe19, 0xfe2f, 0xfe31, 
    0xfe33, 0xfe35, 0xfe44, 0xfe46, 0xfe48, 0xfe4c, 0xfe4f, 0xfe52, 0xfe57, 0xfe59, 0xfe5e, 0xfe61, 0xfe64, 0xfe66, 0xfe6a, 0xfe6b, 0xfe74, 0xfefc, 0xfeff, 0xff03, 
    0xff05, 0xff07, 0xff09, 0xff0c, 0xff0e, 0xff10, 0xff19, 0xff1b, 0xff1e, 0xff20, 0xff3a, 0xff3c, 0xff3e, 0xff40, 0xff5a, 0xff5c, 0xff5e, 0xff60, 0xff62, 0xff64, 
    0xff66, 0xff6f, 0xff71, 0xff9d, 0xff9f, 0xffbe, 0xffc7, 0xffcf, 0xffd7, 0xffdc, 0xffe1, 0xffe3, 0xffe5, 0xffe6, 0xffe9, 0xffec, 0xffee, 0xfffb, 0xfffd, 
)

private val categoryOfRange = intArrayOf(
    0x000f, 0x180c, 0x0018, 0x181a, 0x0018, 0x1615, 0x1918, 0x1418, 0x1809, 0x0009, 0x0018, 0x0019, 0x0018, 0x0001, 0x1518, 0x161b, 0x171b, 0x0002, 0x1519, 0x1619, 
    0x000f, 0x180c, 0x001a, 0x181c, 0x1c1b, 0x1d05, 0x1019, 0x1b1c, 0x190b, 0x0b1b, 0x0218, 0x181b, 0x0b05, 0x1e0b, 0x000b, 0x1801, 0x0001, 0x1901, 0x0001, 0x0002, 
    0x1902, 0x0002, 0x0201, 0x0102, 0x0201, 0x0102, 0x0002, 0x0001, 0x0201, 0x0102, 0x0001, 0x0002, 0x0001, 0x0102, 0x0201, 0x0001, 0x0002, 0x0001, 0x0102, 0x0201, 
    0x0102, 0x0201, 0x0102, 0x0001, 0x0102, 0x0201, 0x0502, 0x0201, 0x0002, 0x0005, 0x0301, 0x0102, 0x0203, 0x0301, 0x0102, 0x0201, 0x0102, 0x0203, 0x0201, 0x0001, 
    0x0201, 0x0002, 0x0001, 0x0102, 0x0201, 0x0102, 0x0001, 0x0201, 0x0002, 0x0205, 0x0002, 0x0004, 0x001b, 0x0004, 0x001b, 0x0004, 0x001b, 0x1b04, 0x001b, 0x0006, 
    0x0201, 0x1b04, 0x0201, 0x0204, 0x0002, 0x0118, 0x001b, 0x1801, 0x0001, 0x0001, 0x0001, 0x0102, 0x0001, 0x0001, 0x0002, 0x0102, 0x0201, 0x0001, 0x0002, 0x0201, 
    0x0002, 0x0201, 0x0119, 0x0102, 0x0201, 0x0102, 0x0001, 0x0002, 0x0201, 0x061c, 0x0006, 0x0007, 0x0201, 0x0102, 0x0201, 0x0001, 0x0418, 0x0018, 0x0002, 0x1814, 
    0x001c, 0x001a, 0x0006, 0x0614, 0x0618, 0x1806, 0x0618, 0x0005, 0x0005, 0x0018, 0x0010, 0x0019, 0x0018, 0x1a18, 0x181c, 0x1c06, 0x0006, 0x1810, 0x0018, 0x0005, 
    0x0504, 0x0005, 0x0006, 0x0009, 0x0018, 0x0005, 0x0506, 0x0005, 0x0518, 0x0006, 0x101c, 0x0006, 0x0004, 0x0006, 0x1c06, 0x0006, 0x0005, 0x0009, 0x0005, 0x001c, 
    0x0518, 0x0018, 0x1005, 0x0605, 0x0005, 0x0006, 0x0005, 0x0006, 0x0005, 0x0009, 0x0005, 0x0006, 0x0004, 0x181c, 0x0018, 0x0004, 0x061a, 0x1a05, 0x0005, 0x0006, 
    0x0604, 0x0006, 0x0604, 0x0006, 0x0604, 0x0006, 0x0018, 0x0005, 0x0006, 0x0018, 0x0005, 0x0005, 0x0005, 0x0006, 0x0610, 0x0006, 0x0805, 0x0005, 0x0806, 0x0508, 
    0x0008, 0x0006, 0x0008, 0x0608, 0x0805, 0x0006, 0x0005, 0x0006, 0x0018, 0x0009, 0x0418, 0x0005, 0x0608, 0x0008, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 
    0x0506, 0x0008, 0x0006, 0x0008, 0x0008, 0x0605, 0x0008, 0x0005, 0x0005, 0x0006, 0x0009, 0x0005, 0x001a, 0x000b, 0x1a1c, 0x1805, 0x0006, 0x0006, 0x0008, 0x0005, 
    0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0006, 0x0008, 0x0006, 0x0006, 0x0006, 0x0006, 0x0005, 0x0005, 0x0009, 0x0006, 0x0005, 0x0618, 0x0006, 0x0008, 
    0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0506, 0x0008, 0x0006, 0x0006, 0x0008, 0x0008, 0x0006, 0x0005, 0x0005, 0x0006, 0x0009, 0x1a18, 0x0506, 0x0006, 
    0x0608, 0x0008, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0506, 0x0608, 0x0006, 0x0008, 0x0008, 0x0006, 0x0006, 0x0008, 0x0005, 0x0005, 0x0006, 0x0009, 
    0x051c, 0x000b, 0x0506, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0008, 0x0806, 0x0008, 0x0008, 0x0008, 0x0006, 0x0005, 0x0008, 
    0x0009, 0x000b, 0x001c, 0x1a1c, 0x0806, 0x0008, 0x0506, 0x0005, 0x0005, 0x0005, 0x0005, 0x0506, 0x0006, 0x0008, 0x0006, 0x0006, 0x0006, 0x0005, 0x0005, 0x0006, 
    0x0009, 0x180b, 0x000b, 0x1c05, 0x0608, 0x0818, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0506, 0x0608, 0x0008, 0x0806, 0x0008, 0x0008, 0x0006, 0x0008, 0x0005, 
    0x0005, 0x0006, 0x0009, 0x0005, 0x0006, 0x0008, 0x0005, 0x0005, 0x0005, 0x0006, 0x0508, 0x0008, 0x0006, 0x0008, 0x0008, 0x0605, 0x001c, 0x0005, 0x080b, 0x000b, 
    0x0005, 0x0006, 0x0009, 0x000b, 0x1c05, 0x0005, 0x0608, 0x0008, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0006, 0x0008, 0x0006, 0x0006, 0x0008, 0x0009, 0x0008, 
    0x0018, 0x0005, 0x0605, 0x0506, 0x0006, 0x1a05, 0x0005, 0x0604, 0x0006, 0x1809, 0x0009, 0x0018, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0605, 0x0506, 
    0x0006, 0x0005, 0x0005, 0x0004, 0x0006, 0x0009, 0x0005, 0x1c05, 0x001c, 0x0018, 0x1c18, 0x001c, 0x0006, 0x001c, 0x0009, 0x000b, 0x061c, 0x1615, 0x0008, 0x0005, 
    0x0005, 0x0006, 0x0806, 0x0006, 0x1806, 0x0605, 0x0005, 0x0006, 0x0006, 0x001c, 0x1c06, 0x001c, 0x001c, 0x0018, 0x001c, 0x0018, 0x0005, 0x0008, 0x0006, 0x0806, 
    0x0006, 0x0608, 0x0806, 0x0608, 0x0506, 0x0009, 0x0018, 0x0005, 0x0008, 0x0006, 0x0005, 0x0006, 0x0508, 0x0008, 0x0005, 0x0008, 0x0005, 0x0006, 0x0005, 0x0806, 
    0x0608, 0x0806, 0x0008, 0x0605, 0x0809, 0x0009, 0x0008, 0x061c, 0x1c01, 0x0001, 0x0001, 0x0001, 0x0002, 0x1804, 0x0002, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 
    0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0006, 0x0018, 0x000b, 0x0005, 0x001c, 0x0001, 0x0002, 0x0514, 0x0005, 
    0x1c18, 0x0005, 0x050c, 0x0005, 0x1516, 0x0005, 0x0018, 0x000a, 0x0005, 0x0005, 0x0005, 0x0006, 0x0005, 0x0006, 0x0018, 0x0005, 0x0006, 0x0005, 0x0005, 0x0006, 
    0x0005, 0x0006, 0x0608, 0x0006, 0x0008, 0x0806, 0x0608, 0x0006, 0x0018, 0x0418, 0x0018, 0x1a05, 0x0006, 0x0009, 0x000b, 0x0018, 0x1814, 0x0018, 0x0006, 0x0010, 
    0x0009, 0x0005, 0x0405, 0x0005, 0x0005, 0x0006, 0x0005, 0x0605, 0x0005, 0x0005, 0x0006, 0x0008, 0x0006, 0x0008, 0x0008, 0x0806, 0x0008, 0x0006, 0x001c, 0x0018, 
    0x0009, 0x0005, 0x0005, 0x0005, 0x0005, 0x0009, 0x000b, 0x001c, 0x0005, 0x0006, 0x0008, 0x0006, 0x0018, 0x0005, 0x0806, 0x0006, 0x0806, 0x0608, 0x0006, 0x0008, 
    0x0006, 0x0609, 0x0009, 0x0009, 0x0018, 0x0418, 0x0018, 0x0006, 0x0607, 0x0006, 0x0006, 0x0508, 0x0005, 0x0806, 0x0006, 0x0806, 0x0008, 0x0806, 0x0508, 0x0005, 
    0x0009, 0x0018, 0x001c, 0x0006, 0x001c, 0x0006, 0x0508, 0x0005, 0x0806, 0x0006, 0x0008, 0x0006, 0x0608, 0x0006, 0x0005, 0x0009, 0x0005, 0x0806, 0x0608, 0x0008, 
    0x0608, 0x0006, 0x0008, 0x0018, 0x0005, 0x0008, 0x0006, 0x0008, 0x0006, 0x0018, 0x0009, 0x0005, 0x0009, 0x0005, 0x0004, 0x0018, 0x0002, 0x0001, 0x0001, 0x0018, 
    0x0006, 0x1806, 0x0006, 0x0806, 0x0006, 0x0005, 0x0605, 0x0005, 0x0506, 0x0805, 0x0006, 0x0005, 0x0002, 0x0004, 0x0002, 0x0204, 0x0002, 0x0004, 0x0006, 0x0006, 
    0x0201, 0x0002, 0x0201, 0x0002, 0x0001, 0x0002, 0x0001, 0x0002, 0x0001, 0x0002, 0x0001, 0x0002, 0x0001, 0x0002, 0x0001, 0x0001, 0x0001, 0x0102, 0x0002, 0x0001, 
    0x0002, 0x0002, 0x0003, 0x0002, 0x0003, 0x0002, 0x0003, 0x0002, 0x0002, 0x0001, 0x1b03, 0x1b02, 0x001b, 0x0002, 0x0002, 0x0001, 0x1b03, 0x001b, 0x0002, 0x0002, 
    0x0001, 0x001b, 0x0002, 0x0001, 0x001b, 0x0002, 0x0002, 0x0001, 0x1b03, 0x001b, 0x000c, 0x0010, 0x0014, 0x0018, 0x1e1d, 0x1d15, 0x1e1d, 0x1d15, 0x0018, 0x0e0d, 
    0x0010, 0x0c18, 0x0018, 0x1d1e, 0x0018, 0x0017, 0x0018, 0x1519, 0x1816, 0x0018, 0x1819, 0x1817, 0x0018, 0x0c10, 0x0010, 0x0010, 0x040b, 0x000b, 0x0019, 0x1516, 
    0x040b, 0x000b, 0x0019, 0x1516, 0x0004, 0x001a, 0x0006, 0x0007, 0x0607, 0x0007, 0x0006, 0x001c, 0x1c01, 0x001c, 0x011c, 0x1c02, 0x0001, 0x0002, 0x0001, 0x021c, 
    0x011c, 0x1c19, 0x0001, 0x001c, 0x1c01, 0x0001, 0x021c, 0x0001, 0x0502, 0x0005, 0x021c, 0x1c02, 0x0201, 0x0119, 0x0019, 0x0102, 0x0002, 0x191c, 0x1c02, 0x000b, 
    0x000a, 0x0102, 0x000a, 0x0b1c, 0x001c, 0x0019, 0x001c, 0x0019, 0x001c, 0x1c19, 0x191c, 0x1c19, 0x001c, 0x1c19, 0x001c, 0x0019, 0x001c, 0x1c19, 0x001c, 0x0019, 
    0x001c, 0x1615, 0x001c, 0x0019, 0x001c, 0x1516, 0x001c, 0x1c19, 0x001c, 0x0019, 0x001c, 0x0019, 0x001c, 0x001c, 0x000b, 0x001c, 0x000b, 0x001c, 0x191c, 0x001c, 
    0x191c, 0x001c, 0x0019, 0x001c, 0x191c, 0x001c, 0x1615, 0x000b, 0x001c, 0x0019, 0x1516, 0x0019, 0x1615, 0x0019, 0x001c, 0x0019, 0x1516, 0x0019, 0x1615, 0x0019, 
    0x1615, 0x0019, 0x001c, 0x0019, 0x001c, 0x0019, 0x001c, 0x001c, 0x001c, 0x0001, 0x0002, 0x0201, 0x0001, 0x0002, 0x0102, 0x0001, 0x0201, 0x0102, 0x0002, 0x0004, 
    0x0001, 0x0201, 0x1c02, 0x001c, 0x0102, 0x0006, 0x0201, 0x0018, 0x0b18, 0x1802, 0x0002, 0x0002, 0x0002, 0x0005, 0x0418, 0x0605, 0x0005, 0x0005, 0x0005, 0x0005, 
    0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0006, 0x0018, 0x1e1d, 0x0018, 0x1d1e, 0x181d, 0x1e18, 0x0018, 0x1418, 0x1814, 0x1e1d, 0x0018, 0x1e1d, 0x1615, 0x0018, 
    0x0418, 0x0018, 0x0014, 0x0018, 0x1814, 0x1815, 0x0018, 0x001c, 0x0018, 0x001c, 0x001c, 0x001c, 0x001c, 0x180c, 0x0018, 0x041c, 0x0a05, 0x1615, 0x001c, 0x1615, 
    0x1514, 0x0016, 0x0a1c, 0x000a, 0x0006, 0x0008, 0x0414, 0x0004, 0x001c, 0x000a, 0x0405, 0x181c, 0x001c, 0x0005, 0x0006, 0x001b, 0x0004, 0x0514, 0x0005, 0x1804, 
    0x0004, 0x0005, 0x0005, 0x0005, 0x001c, 0x000b, 0x001c, 0x0005, 0x001c, 0x0005, 0x001c, 0x000b, 0x001c, 0x000b, 0x0b1c, 0x000b, 0x001c, 0x000b, 0x001c, 0x000b, 
    0x001c, 0x0005, 0x001c, 0x0005, 0x0005, 0x0405, 0x0005, 0x001c, 0x0005, 0x0004, 0x0018, 0x0005, 0x1804, 0x0018, 0x0005, 0x0009, 0x0005, 0x0201, 0x0605, 0x0007, 
    0x1806, 0x0006, 0x0418, 0x0201, 0x0004, 0x0006, 0x0005, 0x000a, 0x0006, 0x0018, 0x001b, 0x0004, 0x001b, 0x0201, 0x0002, 0x0201, 0x0204, 0x0002, 0x0102, 0x0201, 
    0x1b04, 0x011b, 0x0102, 0x0501, 0x0201, 0x0002, 0x0201, 0x0001, 0x0201, 0x0001, 0x0201, 0x0201, 0x0001, 0x0102, 0x0102, 0x0504, 0x0402, 0x0005, 0x0506, 0x0005, 
    0x0506, 0x0005, 0x0605, 0x0005, 0x0008, 0x0006, 0x081c, 0x001c, 0x0006, 0x000b, 0x001c, 0x1c1a, 0x0005, 0x0018, 0x0008, 0x0005, 0x0008, 0x0006, 0x0018, 0x0009, 
    0x0006, 0x0005, 0x0018, 0x0518, 0x0605, 0x0009, 0x0005, 0x0006, 0x0018, 0x0005, 0x0006, 0x0008, 0x1805, 0x0005, 0x0006, 0x0805, 0x0005, 0x0608, 0x0806, 0x0006, 
    0x0008, 0x0006, 0x0008, 0x0018, 0x0409, 0x0009, 0x0018, 0x0005, 0x0604, 0x0005, 0x0009, 0x0005, 0x0005, 0x0006, 0x0008, 0x0006, 0x0008, 0x0006, 0x0005, 0x0605, 
    0x0005, 0x0806, 0x0009, 0x0018, 0x0005, 0x0504, 0x0005, 0x001c, 0x0805, 0x0806, 0x0005, 0x0506, 0x0006, 0x0005, 0x0006, 0x0005, 0x0006, 0x0605, 0x0005, 0x0418, 
    0x1805, 0x0005, 0x0806, 0x0608, 0x0818, 0x1805, 0x0004, 0x0806, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0002, 0x1b04, 0x0004, 0x0002, 0x041b, 0x001b, 0x0002, 
    0x0005, 0x0008, 0x0608, 0x0806, 0x1808, 0x0006, 0x0009, 0x0005, 0x0005, 0x0005, 0x0013, 0x0013, 0x0013, 0x0012, 0x0005, 0x0005, 0x0002, 0x0002, 0x0506, 0x0005, 
    0x1905, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x001b, 0x0005, 0x1516, 0x0005, 0x0005, 0x0005, 0x1c1a, 0x0006, 0x0018, 0x1516, 0x0018, 0x0006, 0x1418, 
    0x1714, 0x1517, 0x1516, 0x0018, 0x1516, 0x0018, 0x0017, 0x0018, 0x0018, 0x1514, 0x1516, 0x0018, 0x1419, 0x0019, 0x1a18, 0x0018, 0x0005, 0x0005, 0x0010, 0x0018, 
    0x181a, 0x0018, 0x1615, 0x1918, 0x1418, 0x1809, 0x0009, 0x0018, 0x0019, 0x0018, 0x0001, 0x1518, 0x161b, 0x171b, 0x0002, 0x1519, 0x1619, 0x1516, 0x1815, 0x1618, 
    0x1805, 0x0005, 0x0504, 0x0005, 0x0004, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x001a, 0x1b19, 0x1a1c, 0x001a, 0x191c, 0x0019, 0x001c, 0x0010, 0x001c, 
)

private fun binarySearchRange(array: IntArray, needle: Int): Int {
    var bottom = 0
    var top = array.size - 1
    var middle = -1
    var value = 0
    while (bottom <= top) {
        middle = (bottom + top) / 2
        value = array[middle]
        if (needle > value)
            bottom = middle + 1
        else if (needle == value)
            return middle
        else
            top = middle - 1
    }
    return middle - (if (needle < value) 1 else 0)
}

internal fun getCategoryValue(ch: Int): Int {
    val index = binarySearchRange(rangeStart, ch)
    val high = rangeEnd[index]
    if (ch <= high) {
        val code = categoryOfRange[index]
        if (code < 0x100) {
            return code
        }
        return if ((ch and 1) == 1) code shr 8 else code and 0xff
    }
    return CharCategory.UNASSIGNED.value
}
