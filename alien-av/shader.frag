/////////////////////////start Pd Header
uniform vec3 iResolution;
uniform float iTime;
uniform float iGlobalTime;
uniform vec4 iMouse;
uniform float a, b, c, d, e, f, g, h;

void mainImage(out vec4 fragColor, in vec2 fragCoord);

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
/////////////////////////end Pd Header

// ============================================================
// 4.frag — Ridged Terrain Mesh Sphere
// ============================================================
// PARAM MAP (all inputs 0.0–1.0 from Pd):
//   a = Distortion — noise displacement on sphere surface
//   b = Grid density — wireframe mesh line count
//   c = Sharpness — ridge peak sharpness of terrain texture
//   d = Rotation speed — rate of sphere rotation
//   e = Terrain scale — spatial zoom of ridged terrain on surface
//   f = Erosion — blend erosion detail into terrain texture
//   g = Pulse — rhythmic expansion/contraction
//   h = Mesh mix — blend between solid terrain and wireframe overlay
// ============================================================

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3 saturate3(vec3 x) { return clamp(x, 0.0, 1.0); }
float param(float p, float lo, float hi) { return mix(lo, hi, p); }

#define PI 3.14159265359
#define TAU 6.28318530718

// --- 3D Hash / Noise ---
float hash3(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 fr = fract(p);
    fr = fr * fr * (3.0 - 2.0 * fr);
    return mix(mix(mix(hash3(i + vec3(0,0,0)), hash3(i + vec3(1,0,0)), fr.x),
                   mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), fr.x), fr.y),
               mix(mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), fr.x),
                   mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), fr.x), fr.y), fr.z);
}

float fbm3(vec3 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 4; i++) {
        v += amp * vnoise(p);
        p = p * 2.01 + 0.5;
        amp *= 0.5;
    }
    return v;
}

// --- 2D Hash / Noise (for terrain) ---
float hash2(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise2(vec2 p) {
    vec2 i = floor(p);
    vec2 fr = fract(p);
    fr = fr * fr * (3.0 - 2.0 * fr);
    float aa = hash2(i);
    float bb = hash2(i + vec2(1.0, 0.0));
    float cc = hash2(i + vec2(0.0, 1.0));
    float dd = hash2(i + vec2(1.0, 1.0));
    return mix(mix(aa, bb, fr.x), mix(cc, dd, fr.x), fr.y);
}

float fbm2(vec2 p, int octaves) {
    float v = 0.0, amp = 0.5;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        v += amp * vnoise2(p);
        p = rot * p * 2.0 + 100.0;
        amp *= 0.5;
    }
    return v;
}

// --- Ridged noise / FBM ---
float ridgedNoise(vec2 p) {
    return 1.0 - abs(vnoise2(p) * 2.0 - 1.0);
}

float ridgedFBM(vec2 p, float sharpness) {
    float v = 0.0, amp = 0.5, freq = 1.0, prev = 1.0;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 6; i++) {
        float n = ridgedNoise(p * freq);
        n = pow(n, sharpness);
        v += n * amp * prev;
        prev = n;
        freq *= 2.0;
        amp *= 0.5;
        p = rot * p;
    }
    return v;
}

// --- Rotation ---
mat3 rotY(float ang) { float c = cos(ang), s = sin(ang); return mat3(c,0,s, 0,1,0, -s,0,c); }
mat3 rotX(float ang) { float c = cos(ang), s = sin(ang); return mat3(1,0,0, 0,c,-s, 0,s,c); }

// --- Sphere SDF with distortion ---
float sphereSDF(vec3 p, float radius, float distAmt, float t) {
    float base = length(p) - radius;
    vec3 n = normalize(p);
    float disp = fbm3(n * 4.0 + t * 0.3) * 2.0 - 1.0;
    return base - disp * distAmt;
}

// --- Mesh wireframe pattern ---
float meshPattern(vec3 p, float density, float lineW) {
    vec3 n = normalize(p);
    float theta = acos(clamp(n.y, -1.0, 1.0));
    float phi = atan(n.z, n.x);

    float gridTheta = sin(theta * density) * 0.5 + 0.5;
    float gridPhi = sin(phi * density) * 0.5 + 0.5;

    float lineTheta = smoothstep(lineW, 0.0, abs(gridTheta - 0.5) * 2.0);
    float linePhi = smoothstep(lineW, 0.0, abs(gridPhi - 0.5) * 2.0);

    float mesh = max(lineTheta, linePhi);

    float diag = sin((theta + phi) * density * 0.7);
    float lineDiag = smoothstep(lineW, 0.0, abs(diag) * 1.5);
    mesh = max(mesh, lineDiag * 0.7);

    return mesh;
}

// --- Terrain texture mapped to sphere surface ---
float terrainTexture(vec3 p, float terrainScale, float sharpness, float erosion, float t) {
    vec3 n = normalize(p);
    float theta = acos(clamp(n.y, -1.0, 1.0));
    float phi = atan(n.z, n.x);

    // Map sphere coords to 2D for terrain
    vec2 tc = vec2(phi / TAU + 0.5, theta / PI) * terrainScale;
    tc += vec2(t * 0.05, t * 0.03);

    float terrain = ridgedFBM(tc, sharpness);

    // Erosion detail
    float eroded = fbm2(tc * 2.0, 5);
    terrain = mix(terrain, terrain * eroded, erosion);

    return terrain;
}

// --- Raymarching ---
float raymarch(vec3 ro, vec3 rd, float radius, float distAmt, float t) {
    float dist = 0.0;
    for (int i = 0; i < 80; i++) {
        vec3 p = ro + rd * dist;
        float sd = sphereSDF(p, radius, distAmt, t);
        if (abs(sd) < 0.001) break;
        if (dist > 5.0) break;
        dist += sd * 0.7;
    }
    return dist;
}

// --- Normal estimation ---
vec3 calcNormal(vec3 p, float radius, float distAmt, float t) {
    vec2 ep = vec2(0.002, 0.0);
    return normalize(vec3(
        sphereSDF(p + ep.xyy, radius, distAmt, t) - sphereSDF(p - ep.xyy, radius, distAmt, t),
        sphereSDF(p + ep.yxy, radius, distAmt, t) - sphereSDF(p - ep.yxy, radius, distAmt, t),
        sphereSDF(p + ep.yyx, radius, distAmt, t) - sphereSDF(p - ep.yyx, radius, distAmt, t)
    ));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    float distAmt      = param(a, 0.0, 0.6);
    float density       = param(b, 8.0, 40.0);
    float sharpness     = param(c, 0.5, 3.0);
    float rotSpeed      = param(d, 0.1, 1.5);
    float terrainScale  = param(e, 2.0, 10.0);
    float erosion       = param(f, 0.0, 1.0);
    float pulseAmt      = param(g, 0.0, 0.3);
    float meshMix       = param(h, 0.0, 1.0);

    float t = iTime;
    float radius = 0.8 + sin(t * 2.0) * pulseAmt;

    // Camera
    vec3 ro = vec3(0.0, 0.0, 2.5);
    vec3 rd = normalize(vec3(uv, -1.0));

    // Rotate
    mat3 rot = rotY(t * rotSpeed) * rotX(t * rotSpeed * 0.4);
    vec3 ro2 = rot * ro;
    vec3 rd2 = rot * rd;

    float dist = raymarch(ro2, rd2, radius, distAmt, t);

    vec3 col = vec3(0.0);

    if (dist < 5.0) {
        vec3 p = ro2 + rd2 * dist;
        vec3 nor = calcNormal(p, radius, distAmt, t);

        // Terrain texture on surface
        float terrain = terrainTexture(p, terrainScale, sharpness, erosion, t);

        // Wireframe mesh overlay
        float mesh = meshPattern(p, density, 0.15);

        // Lighting
        vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
        float diff = max(dot(nor, lightDir), 0.0);
        float fres = pow(1.0 - max(dot(nor, -normalize(rd2)), 0.0), 3.0);

        // Combine terrain fill with mesh wireframe
        float terrainLit = terrain * (0.3 + 0.7 * diff);
        float meshLine = mesh * (0.5 + 0.5 * diff);

        // meshMix: 0 = full terrain, 1 = terrain visible only through wireframe
        float surface = mix(terrainLit, terrainLit * 0.3 + meshLine * terrainLit * 0.7, meshMix);
        surface += fres * 0.2;

        col = vec3(surface);
    }

    // Background glow
    float bgGlow = exp(-length(uv) * 2.0) * 0.04;
    col += bgGlow;

    col = saturate3(col);
    fragColor = vec4(col, 1.0);
}
