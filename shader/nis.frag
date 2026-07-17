// NVIDIA Image Scaling SDK v1.0.3 - adapted to fragment shader
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Adapted from the NIS compute shader to a single-pass fragment shader for mkxp-z.
// Coefficient tables are uploaded as uniform arrays from C++.

#ifdef GLSLES
    precision highp float;
#endif

uniform sampler2D texture;
uniform vec2 sourceSize;
uniform vec2 targetScale;
uniform float sharpness;

// NIS coefficient tables: 64 phases x 6 taps, flattened to 1D (384 elements each)
uniform float coef_scale[384];
uniform float coef_usm[384];

varying vec2 v_texCoord;

#define kPhaseCount 64
#define kFilterSize 6
#define NIS_SCALE_FLOAT 1.0
#define NIS_SCALE_INT 1

#define kDetectRatio (2.0 * 1127.0 / 1024.0)
#define kDetectThres (64.0 / 1024.0)
#define kMinContrastRatio 2.0
#define kRatioNorm (1.0 / (10.0 - kMinContrastRatio))
#define kContrastBoost 1.0
#define kEps (1.0 / 255.0)
#define kSharpStartY 0.45
#define kSharpScaleY (1.0 / (0.9 - kSharpStartY))

#define saturate(x) clamp(x, 0.0, 1.0)
#define lerp(a, b, x) mix(a, b, x)

float getY(vec3 rgba)
{
    return 0.2126 * rgba.x + 0.7152 * rgba.y + 0.0722 * rgba.z;
}

// Edge map computation from a 6x6 luma neighborhood (flattened, row-major: idx = row*6+col)
vec4 GetEdgeMap(float p[36], int baseI, int baseJ)
{
    float g_0 = abs(
        p[(baseJ+0)*6+baseI+0] + p[(baseJ+0)*6+baseI+1] + p[(baseJ+0)*6+baseI+2]
        - p[(baseJ+2)*6+baseI+0] - p[(baseJ+2)*6+baseI+1] - p[(baseJ+2)*6+baseI+2]
    );
    float g_45 = abs(
        p[(baseJ+1)*6+baseI+0] + p[(baseJ+0)*6+baseI+0] + p[(baseJ+0)*6+baseI+1]
        - p[(baseJ+2)*6+baseI+1] - p[(baseJ+2)*6+baseI+2] - p[(baseJ+1)*6+baseI+2]
    );
    float g_90 = abs(
        p[(baseJ+0)*6+baseI+0] + p[(baseJ+1)*6+baseI+0] + p[(baseJ+2)*6+baseI+0]
        - p[(baseJ+0)*6+baseI+2] - p[(baseJ+1)*6+baseI+2] - p[(baseJ+2)*6+baseI+2]
    );
    float g_135 = abs(
        p[(baseJ+1)*6+baseI+0] + p[(baseJ+2)*6+baseI+0] + p[(baseJ+2)*6+baseI+1]
        - p[(baseJ+0)*6+baseI+1] - p[(baseJ+0)*6+baseI+2] - p[(baseJ+1)*6+baseI+2]
    );

    float g_0_90_max = max(g_0, g_90);
    float g_0_90_min = min(g_0, g_90);
    float g_45_135_max = max(g_45, g_135);
    float g_45_135_min = min(g_45, g_135);

    float e_0_90 = 0.0;
    float e_45_135 = 0.0;

    if (g_0_90_max + g_45_135_max == 0.0) {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }

    e_0_90 = min(g_0_90_max / (g_0_90_max + g_45_135_max), 1.0);
    e_45_135 = 1.0 - e_0_90;

    bool c_0_90 = (g_0_90_max > (g_0_90_min * kDetectRatio)) && (g_0_90_max > kDetectThres) && (g_0_90_max > g_45_135_min);
    bool c_45_135 = (g_45_135_max > (g_45_135_min * kDetectRatio)) && (g_45_135_max > kDetectThres) && (g_45_135_max > g_0_90_min);
    bool c_g_0_90 = g_0_90_max == g_0;
    bool c_g_45_135 = g_45_135_max == g_45;

    float f_e_0_90 = (c_0_90 && c_45_135) ? e_0_90 : 1.0;
    float f_e_45_135 = (c_0_90 && c_45_135) ? e_45_135 : 1.0;

    float weight_0 = (c_0_90 && c_g_0_90) ? f_e_0_90 : 0.0;
    float weight_90 = (c_0_90 && !c_g_0_90) ? f_e_0_90 : 0.0;
    float weight_45 = (c_45_135 && c_g_45_135) ? f_e_45_135 : 0.0;
    float weight_135 = (c_45_135 && !c_g_45_135) ? f_e_45_135 : 0.0;

    return vec4(weight_0, weight_90, weight_45, weight_135);
}

float CalcLTI(float p0, float p1, float p2, float p3, float p4, float p5, int phase_index)
{
    bool selector = (phase_index <= kPhaseCount / 2);
    float sel = selector ? p0 : p3;
    float a_min = min(min(p1, p2), sel);
    float a_max = max(max(p1, p2), sel);
    sel = selector ? p2 : p5;
    float b_min = min(min(p3, p4), sel);
    float b_max = max(max(p3, p4), sel);

    float a_cont = a_max - a_min;
    float b_cont = b_max - b_min;

    float cont_ratio = max(a_cont, b_cont) / (min(a_cont, b_cont) + kEps);
    return (1.0 - saturate((cont_ratio - kMinContrastRatio) * kRatioNorm)) * kContrastBoost;
}

float EvalPoly6(float pxl[6], int phase_int,
                float sharpStrengthMin, float sharpStrengthScale,
                float sharpLimitMin, float sharpLimitScale)
{
    float y = 0.0;
    for (int i = 0; i < 6; i++) {
        y += coef_scale[phase_int * 6 + i] * pxl[i];
    }
    float y_usm = 0.0;
    for (int i = 0; i < 6; i++) {
        y_usm += coef_usm[phase_int * 6 + i] * pxl[i];
    }

    float y_scale = 1.0 - saturate((y * (1.0 / NIS_SCALE_FLOAT) - kSharpStartY) * kSharpScaleY);
    float y_sharpness = y_scale * sharpStrengthScale + sharpStrengthMin;
    y_usm *= y_sharpness;

    float y_sharpness_limit = (y_scale * sharpLimitScale + sharpLimitMin) * y;
    y_usm = min(y_sharpness_limit, max(-y_sharpness_limit, y_usm));
    y_usm *= CalcLTI(pxl[0], pxl[1], pxl[2], pxl[3], pxl[4], pxl[5], phase_int);

    return y + y_usm;
}

float FilterNormal(float p[36], int phase_x, int phase_y)
{
    float h_acc = 0.0;
    for (int j = 0; j < 6; j++) {
        float v_acc = 0.0;
        for (int i = 0; i < 6; i++) {
            v_acc += p[i * 6 + j] * coef_scale[phase_y * 6 + i];
        }
        h_acc += v_acc * coef_scale[phase_x * 6 + j];
    }
    return h_acc;
}

float AddDirFilters(float p[36], float phase_x, float phase_y, int phase_x_int, int phase_y_int, vec4 w,
                    float ssMin, float ssScale, float slMin, float slScale)
{
    float f = 0.0;

    // 0 degree filter (horizontal)
    if (w.x > 0.0) {
        float interp0[6];
        for (int i = 0; i < 6; i++) {
            interp0[i] = lerp(p[i*6+2], p[i*6+3], phase_x);
        }
        f += EvalPoly6(interp0, phase_y_int, ssMin, ssScale, slMin, slScale) * w.x;
    }
    // 90 degree filter (vertical)
    if (w.y > 0.0) {
        float interp90[6];
        for (int i = 0; i < 6; i++) {
            interp90[i] = lerp(p[2*6+i], p[3*6+i], phase_y);
        }
        f += EvalPoly6(interp90, phase_x_int, ssMin, ssScale, slMin, slScale) * w.y;
    }
    // 45 degree filter
    if (w.z > 0.0) {
        float pphase_b45 = 0.5 + 0.5 * (phase_x - phase_y);
        float tmp45[7];
        tmp45[1] = lerp(p[2*6+1], p[1*6+2], pphase_b45);
        tmp45[3] = lerp(p[3*6+2], p[2*6+3], pphase_b45);
        tmp45[5] = lerp(p[4*6+3], p[3*6+4], pphase_b45);
        pphase_b45 = pphase_b45 - 0.5;
        float a45 = (pphase_b45 >= 0.0) ? p[0*6+2] : p[2*6+0];
        float b45 = (pphase_b45 >= 0.0) ? p[1*6+3] : p[3*6+1];
        float c45 = (pphase_b45 >= 0.0) ? p[2*6+4] : p[4*6+2];
        float d45 = (pphase_b45 >= 0.0) ? p[3*6+5] : p[5*6+3];
        tmp45[0] = lerp(p[1*6+1], a45, abs(pphase_b45));
        tmp45[2] = lerp(p[2*6+2], b45, abs(pphase_b45));
        tmp45[4] = lerp(p[3*6+3], c45, abs(pphase_b45));
        tmp45[6] = lerp(p[4*6+4], d45, abs(pphase_b45));

        float interp45[6];
        float pphase_p45 = phase_x + phase_y;
        if (pphase_p45 >= 1.0) {
            for (int i = 0; i < 6; i++) interp45[i] = tmp45[i + 1];
            pphase_p45 = pphase_p45 - 1.0;
        } else {
            for (int i = 0; i < 6; i++) interp45[i] = tmp45[i];
        }
        f += EvalPoly6(interp45, int(pphase_p45 * 64.0), ssMin, ssScale, slMin, slScale) * w.z;
    }
    // 135 degree filter
    if (w.w > 0.0) {
        float pphase_b135 = 0.5 * (phase_x + phase_y);
        float tmp135[7];
        tmp135[1] = lerp(p[3*6+1], p[4*6+2], pphase_b135);
        tmp135[3] = lerp(p[2*6+2], p[3*6+3], pphase_b135);
        tmp135[5] = lerp(p[1*6+3], p[2*6+4], pphase_b135);
        pphase_b135 = pphase_b135 - 0.5;
        float a135 = (pphase_b135 >= 0.0) ? p[5*6+2] : p[3*6+0];
        float b135 = (pphase_b135 >= 0.0) ? p[4*6+3] : p[2*6+1];
        float c135 = (pphase_b135 >= 0.0) ? p[3*6+4] : p[1*6+2];
        float d135 = (pphase_b135 >= 0.0) ? p[2*6+5] : p[0*6+3];
        tmp135[0] = lerp(p[4*6+1], a135, abs(pphase_b135));
        tmp135[2] = lerp(p[3*6+2], b135, abs(pphase_b135));
        tmp135[4] = lerp(p[2*6+3], c135, abs(pphase_b135));
        tmp135[6] = lerp(p[1*6+4], d135, abs(pphase_b135));

        float interp135[6];
        float pphase_p135 = 1.0 + (phase_x - phase_y);
        if (pphase_p135 >= 1.0) {
            for (int i = 0; i < 6; i++) interp135[i] = tmp135[i + 1];
            pphase_p135 = pphase_p135 - 1.0;
        } else {
            for (int i = 0; i < 6; i++) interp135[i] = tmp135[i];
        }
        f += EvalPoly6(interp135, int(pphase_p135 * 64.0), ssMin, ssScale, slMin, slScale) * w.w;
    }
    return f;
}

void main()
{
    vec2 texSizeInv = vec2(1.0) / sourceSize;

    // Compute sharpness-derived parameters
    float sharpen_slider = sharpness - 0.5;
    float MaxScale = (sharpen_slider >= 0.0) ? 1.25 : 1.75;
    float MinScale = (sharpen_slider >= 0.0) ? 1.25 : 1.0;
    float LimitScale = (sharpen_slider >= 0.0) ? 1.25 : 1.0;
    float ssMin = max(0.0, 0.4 + sharpen_slider * MinScale * 1.2);
    float ssMax = 1.6 + sharpen_slider * MaxScale * 1.8;
    float ssScale = ssMax - ssMin;
    float slMin = max(0.1, 0.14 + sharpen_slider * LimitScale * 0.32);
    float slMax = 0.5 + sharpen_slider * LimitScale * 0.6;
    float slScale = slMax - slMin;

    // Source texture position in pixel coordinates (texel-center convention)
    float srcX = v_texCoord.x * sourceSize.x - 0.5;
    float srcY = v_texCoord.y * sourceSize.y - 0.5;

    // Integer and fractional parts
    int px = int(floor(srcX));
    int py = int(floor(srcY));
    float fx = srcX - floor(srcX);
    float fy = srcY - floor(srcY);
    int fx_int = int(min(fx * 64.0, 63.0));
    int fy_int = int(min(fy * 64.0, 63.0));

    // Load 6x6 luma neighborhood (centered 2 texels before current texel)
    float baseX = float(px) - 2.0 + 0.5;
    float baseY = float(py) - 2.0 + 0.5;
    float p[36];
    for (int j = 0; j < 6; j++) {
        for (int i = 0; i < 6; i++) {
            vec2 tc = vec2(baseX + float(i), baseY + float(j)) * texSizeInv;
            p[j * 6 + i] = getY(texture2D(texture, tc).rgb);
        }
    }

    // Compute edge maps for the 2x2 neighborhood around the fractional position.
    // The 4x4 sub-blocks start at grid offset (1,1) to center on the current pixel.
    vec4 edge00 = GetEdgeMap(p, 1, 1);
    vec4 edge01 = GetEdgeMap(p, 2, 1);
    vec4 edge10 = GetEdgeMap(p, 1, 2);
    vec4 edge11 = GetEdgeMap(p, 2, 2);

    // Interpolate edge map weights
    vec4 h0 = lerp(edge00, edge01, fx);
    vec4 h1 = lerp(edge10, edge11, fx);
    vec4 w = lerp(h0, h1, fy) * float(NIS_SCALE_INT);

    // Normal bidirectional filter weight
    float baseWeight = NIS_SCALE_FLOAT - w.x - w.y - w.z - w.w;

    // Combined filter output
    float opY = FilterNormal(p, fx_int, fy_int) * baseWeight;
    opY += AddDirFilters(p, fx, fy, fx_int, fy_int, w, ssMin, ssScale, slMin, slScale);

    // Bilinear chroma tap
    vec2 chromaCoord = vec2(srcX + 0.5, srcY + 0.5) * texSizeInv;
    vec4 op = texture2D(texture, chromaCoord);
    float y_orig = getY(op.rgb);

    // Apply luma correction
    float corr = opY * (1.0 / NIS_SCALE_FLOAT) - y_orig;
    op.x += corr;
    op.y += corr;
    op.z += corr;

    gl_FragColor = op;
}
