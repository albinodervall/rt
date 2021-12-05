#if 0
    clang -Wall -Werror -Wno-unused -O3 -g rt.c -o rt
    exit
#endif

#include <stdlib.h>
#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#include <time.h> // clock_gettime_nsec_np
#include <sys/random.h> // getentropy
#include <pthread.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

#define F32_INFINITY INFINITY
#define F32_PI 3.141592653589793f
#define F32_PI_OVER_2 1.57079632679490f
#define F32_2_PI 6.283185307179586f

typedef struct f32x4 {
    f32 x;
    f32 y;
    f32 z;
} f32x4;

enum Material_Type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
};

struct Spheres {
    f32 x0[488];
    f32 y0[488];
    f32 z0[488];
    f32 x1[488];
    f32 y1[488];
    f32 z1[488];
    f32 r[488];
    struct {
        u32 type[488];
        struct {
            f32 r[488];
            f32 g[488];
            f32 b[488];
        } albedo;
        struct {
            f32 roughness[488];
        } metal;

        struct {
            f32 refractive_index[488];
        } dielectric;

        u32 use_texture[488];
    } material;
};

struct Job {
    u32 from_x;
    u32 from_y;
    u32 to_x;
    u32 to_y;

    u32 total_width;
    u32 total_height;
    u8 * buffer;

    u32 samples_count;
    f32 lens_radius;
    f32x4 look_u;
    f32x4 look_v;
    f32x4 origin;
    f32x4 horizontal;
    f32x4 vertical;
    f32 t0;
    f32 t1;

    struct Spheres * spheres;
    u32 spheres_count;
};

struct Job_Queue {
    struct Job * jobs;
    u32 jobs_count;
    u32 jobs_index;
};

#define SIMD 0
#if SIMD
    #include <arm_neon.h>

    #define SIMD_WIDTH 4

    typedef struct {
        uint32x4_t x;
    } u32x;

    typedef struct {
        float32x4_t x;
    } f32x;

    static inline f32x F32NX(f32 a) { return  ((f32x){ vdupq_n_f32(a) }); }
    static inline u32x U32NX(u32 a) { return  ((u32x){ vdupq_n_u32(a) }); }

    static inline f32x U32X_TO_F32X(u32x a) { return ((f32x){ vcvtq_f32_u32((a).x) }); }
    static inline u32x F32X_TO_U32X(f32x a) { return ((u32x){ vcvtmq_u32_f32((a).x) }); }

    static inline f32x ABSX(f32x a) { return ((f32x){ vabsq_f32((a).x) }); }
    static inline f32x MINX(f32x a, f32x b) { return ((f32x){ vminq_f32((a).x, (b).x) }); }
    static inline f32x MAXX(f32x a, f32x b) { return ((f32x){ vmaxq_f32((a).x, (b).x) }); }

    static inline f32x NEGX(f32x a) { return ((f32x){ vnegq_f32((a).x) }); }
    f32x ADDX_F32(f32x a, f32x b) { return (f32x){ vaddq_f32(a.x, b.x) }; }
    u32x ADDX_U32(u32x a, u32x b) { return (u32x){ vaddq_u32(a.x, b.x) }; }
    #define ADDX(a, b) _Generic((a, b),\
        f32x: ADDX_F32,\
        u32x: ADDX_U32\
    )(a, b)
    static inline f32x MULX(f32x a, f32x b) { return ((f32x){ vmulq_f32((a).x, (b).x) }); }
    static inline f32x SUBX(f32x a, f32x b) { return ((f32x){ vsubq_f32((a).x, (b).x) }); }
    static inline f32x DIVX(f32x a, f32x b) { return ((f32x){ vdivq_f32((a).x, (b).x) }); }
    static inline f32x FMAX(f32x a, f32x b, f32x const c) { return ((f32x){ vfmaq_f32((c).x, (a).x, (b).x) }); }
    // #define WFMS(a, b, c) (f32x){ ((a.x) * (b.x)) - (c.x) }
    static inline f32x SQRTX(f32x a) { return ((f32x){ vsqrtq_f32((a).x) }); }
    static inline f32x RSQRTX(f32x a) { return ((f32x){ vrsqrteq_f32((a).x) }); }

    static inline f32 HADDX_F32(f32x a) { return vaddvq_f32((a).x); }
    static inline u32 HADDX_U32(u32x a) { return vaddvq_u32((a).x); }
    #define HADDX(a) _Generic((a),\
        f32x: HADDX_F32,\
        u32x: HADDX_U32\
    )(a)

    static inline u32x ANDX(u32x a, u32x b) { return ((u32x){ vandq_u32((a).x, (b).x) }); }
    static inline u32x NOTX(u32x a) { return ((u32x){ vmvnq_u32((a).x) }); }
    static inline u32x ORX(u32x a, u32x b) { return ((u32x){ vorrq_u32((a).x, (b).x) }); }
    static inline u32x XORX(u32x a, u32x b) { return ((u32x){ veorq_u32((a).x, (b).x) }); }
    // vshlq_n_u32 and vshrq_n_u32 requires compile-time constants, which means ´
    // clang does not allow us to implement these as functions.
    #define SLLX(a, b) ((u32x){ vshlq_n_u32((a).x, b) })
    #define SRLX(a, b) ((u32x){ vshrq_n_u32((a).x, b) })

    static inline u32x LTX(f32x a, f32x b) { return ((u32x){ vcltq_f32((a).x, (b).x) }); }
    static inline u32x LEX(f32x a, f32x b) { return ((u32x){ vcleq_f32((a).x, (b).x) }); }
    static inline u32x GTX(f32x a, f32x b) { return ((u32x){ vcgtq_f32((a).x, (b).x) }); }
    static inline u32x GEX(f32x a, f32x b) { return ((u32x){ vcgeq_f32((a).x, (b).x) }); }
    static inline u32x EQX(u32x a, u32x b) { return ((u32x){ vceqq_u32((a).x, (b).x) }); }
    static inline u32x NEQX(u32x a, u32x b) { return ((u32x){ vmvnq_u32(vceqq_u32((a).x, (b).x)) }); }
    static inline u32 ALL_ZEROS_X(u32x a) { return (vmaxvq_u32((a).x) == 0); }
    static inline u32 ALL_ONES_X(u32x a) { return (vminvq_u32((a).x) == 0xFFFFFFFF); }

    static inline f32x CHOOSEX_F32(f32x a, f32x b, u32x t) { return ((f32x){ vbslq_f32((t).x, (b).x, (a).x) }); }
    static inline u32x CHOOSEX_U32(u32x a, u32x b, u32x t) { return ((u32x){ vbslq_f32((t).x, (b).x, (a).x) }); }
    #define CHOOSEX(a, b, t) _Generic((a, b),\
        f32x: CHOOSEX_F32,\
        u32x: CHOOSEX_U32\
    )(a, b, t)

    // Unfortunately we downshift to scalar here, but these functions could be implemented in NEON
    f32x ACOSX(f32x a) { return (f32x){ vld1q_f32(((f32[]){ acosf(vgetq_lane_f32((a).x, 0)), acosf(vgetq_lane_f32((a).x, 1)), acosf(vgetq_lane_f32((a).x, 2)), acosf(vgetq_lane_f32((a).x, 3)) })) }; }
    f32x ATAN2X(f32x a, f32x b) { return (f32x){ vld1q_f32(((f32[]){ atan2f(vgetq_lane_f32((a).x, 0), vgetq_lane_f32((b).x, 0)), atan2f(vgetq_lane_f32((a).x, 1), vgetq_lane_f32((b).x, 1)), atan2f(vgetq_lane_f32((a).x, 2), vgetq_lane_f32((b).x, 2)), atan2f(vgetq_lane_f32((a).x, 3), vgetq_lane_f32((b).x, 3)) })) }; }
    f32x POW5X(f32x a) { return (f32x){ vmulq_f32((a).x, vmulq_f32((a).x, vmulq_f32((a).x, vmulq_f32((a).x, (a).x)))) }; }

    f32x GATHERX_F32(void const * base, u32 stride, u32x indices) {
        uint32x4_t i = vmulq_u32(vdupq_n_u32(stride), indices.x);
        return (f32x){ vld1q_f32(((f32[]){
            *(f32 *)(((u8 *)base) + vgetq_lane_u32(i, 0)),
            *(f32 *)(((u8 *)base) + vgetq_lane_u32(i, 1)),
            *(f32 *)(((u8 *)base) + vgetq_lane_u32(i, 2)),
            *(f32 *)(((u8 *)base) + vgetq_lane_u32(i, 3))
        })) };
    }

    u32x GATHERX_U32(void const * base, u32 stride, u32x indices) {
        uint32x4_t i = vmulq_u32(vdupq_n_u32(stride), indices.x);
        return (u32x){ vld1q_u32(((u32[]){
            *(u32 *)(((u8 *)base) + vgetq_lane_u32(i, 0)),
            *(u32 *)(((u8 *)base) + vgetq_lane_u32(i, 1)),
            *(u32 *)(((u8 *)base) + vgetq_lane_u32(i, 2)),
            *(u32 *)(((u8 *)base) + vgetq_lane_u32(i, 3))
        })) };
    }
#else
    #define SIMD_WIDTH 1

    typedef struct {
        u32 x;
    } u32x;

    typedef struct {
        f32 x;
    } f32x;

    static inline f32x F32NX(f32 a) { return (f32x){ a }; }
    static inline u32x U32NX(u32 a) { return (u32x){ a }; }

    static inline f32x U32X_TO_F32X(u32x a) { return (f32x){ (f32)a.x }; }
    static inline u32x F32X_TO_U32X(f32x a) { return (u32x){ (u32)a.x }; }

    static inline f32x ABSX(f32x a) { return (f32x){ fabsf(a.x) }; }
    static inline f32x MINX(f32x a, f32x b) { return (f32x){ (a.x < b.x) ? a.x : b.x }; }
    static inline f32x MAXX(f32x a, f32x b) { return (f32x){ (a.x > b.x) ? a.x : b.x }; }

    static inline f32x NEGX(f32x a) { return (f32x){ -a.x }; }
    static inline f32x ADDX_F32(f32x a, f32x b) { return (f32x){ a.x + b.x }; }
    static inline u32x ADDX_U32(u32x a, u32x b) { return (u32x){ a.x + b.x }; }
    #define ADDX(a, b) _Generic((a, b),\
        f32x: ADDX_F32,\
        u32x: ADDX_U32\
    )(a, b)
    static inline f32x MULX(f32x a, f32x b) { return (f32x){ a.x * b.x }; }
    static inline f32x MULNX(f32x a, f32 b) { return (f32x){ a.x * b }; }
    static inline f32x SUBX(f32x a, f32x b) { return (f32x){ a.x - b.x }; }
    static inline f32x DIVX(f32x a, f32x b) { return (f32x){ a.x / b.x }; }
    static inline f32x FMAX(f32x a, f32x b, f32x c) { return (f32x){ (a.x * b.x) + c.x }; } // TODO: Should be a.x + (b.x * c.x) to match NEON
    static inline f32x FMANX(f32x a, f32 b, f32x c) { return (f32x){ (a.x * b) + c.x }; } // TODO: Should be a.x + (b.x * c.x) to match NEON
    // static inline f32x FMSX(f32x a, f32x b, f32x c) { return (f32x){ (a.x * b.x) - c.x }; } // TODO: Should be a.x - (b.x * c.x) to match NEON
    static inline f32x SQRTX(f32x a) { return (f32x){ sqrtf(a.x) }; }
    static inline f32x RSQRTX(f32x a) { return (f32x){ 1.0f / sqrtf(a.x) }; }

    static inline f32 HADDX_F32(f32x a) { return a.x; }
    static inline u32 HADDX_U32(u32x a) { return a.x; }
    #define HADDX(a) _Generic((a),\
        f32x: HADDX_F32,\
        u32x: HADDX_U32\
    )(a)

    static inline u32x ANDX(u32x a, u32x b) { return (u32x){ a.x & b.x }; }
    static inline u32x NOTX(u32x a) { return (u32x){ ~a.x }; }
    static inline u32x ORX(u32x a, u32x b) { return  (u32x){ a.x | b.x }; }
    static inline u32x XORX(u32x a, u32x b) { return  (u32x){ a.x ^ b.x }; }
    static inline u32x SLLX(u32x a, u32 b) { return  (u32x){ a.x << b }; }
    static inline u32x SRLX(u32x a, u32 b) { return  (u32x){ a.x >> b }; }

    static inline u32x LTX(f32x a, f32x b) { return (u32x){ (a.x < b.x) ? 0xFFFFFFFF : 0 }; }
    static inline u32x LEX(f32x a, f32x b) { return (u32x){ (a.x <= b.x) ? 0xFFFFFFFF : 0 }; }
    static inline u32x GTX(f32x a, f32x b) { return (u32x){ (a.x > b.x) ? 0xFFFFFFFF : 0 }; }
    static inline u32x GEX(f32x a, f32x b) { return (u32x){ (a.x >= b.x) ? 0xFFFFFFFF : 0 }; }
    static inline u32x EQX(u32x a, u32x b) { return (u32x){ (a.x == b.x) ? 0xFFFFFFFF : 0 }; }
    static inline u32x NEQX(u32x a, u32x b) { return (u32x){ (a.x != b.x) ? 0xFFFFFFFF : 0 }; }
    static inline u32 ALL_ZEROS_X(u32x a) { return (a.x == 0); }
    static inline u32 ALL_ONES_X(u32x a) { return (a.x == 0xFFFFFFFF); }

    static inline f32x CHOOSEX_F32(f32x a, f32x b, u32x t) { return (f32x){ (t.x == 0) ? a.x : b.x }; }
    static inline u32x CHOOSEX_U32(u32x a, u32x b, u32x t) { return (u32x){ (t.x == 0) ? a.x : b.x }; }
    #define CHOOSEX(a, b, t) _Generic((a, b),\
        f32x: CHOOSEX_F32,\
        u32x: CHOOSEX_U32\
    )(a, b, t)

    #define POW5X(a) ((f32x){ powf((a).x, 5.0f) })

    #define SINCOSX(a, s, c) *s = (f32x){ sinf(a.x) }; *c = (f32x){ cosf(a.x) };
    #define ACOSX(a) ((f32x){ acosf((a).x) })
    #define ATAN2X(a, b) ((f32x){ atan2f((a).x, (b).x) })

    #define GATHERX_F32(base, stride, indices) ((f32x){ *(f32 *)(((u8 *)(base)) + ((stride) * (indices).x)) })
    #define GATHERX_U32(base, stride, indices) ((u32x){ *(u32 *)(((u8 *)(base)) + ((stride) * (indices).x)) })
#endif

typedef struct {
    union {
        f32x x;
        f32x r;
    };

    union {
        f32x y;
        f32x g;
    };

    union {
        f32x z;
        f32x b;
    };
} vf32x;

static inline vf32x VF32X(f32x a, f32x b, f32x c) {
    return (vf32x){ { a }, { b }, { c } };
}

static inline vf32x VF32NX(f32 a, f32 b, f32 c) {
    return (vf32x){ { F32NX(a) }, { F32NX(b) }, { F32NX(c) } };
}

#define VABSX(a) ((vf32x){ { ABSX((a).x) }, { ABSX((a).y) }, { ABSX((a).x) } })
#define VMINX(a, b) ((vf32x){ { MINX((a).x, (b).x) }, { MINX((a).y, (b).y) }, { MINX((a).z, (b).z) } })
#define VMAXX(a, b) ((vf32x){ { MAXX((a).x, (b).x) }, { MAXX((a).y, (b).y) }, { MAXX((a).z, (b).z) } })

#define VNEGX(a) ((vf32x){ { NEGX((a).x) }, { NEGX((a).y) }, { NEGX((a).z) } })
#define VADDX(a, b) ((vf32x){ { ADDX((a).x, (b).x) }, { ADDX((a).y, (b).y) }, { ADDX((a).z, (b).z) } })
#define VMULX(a, b) ((vf32x){ { MULX((a).x, (b).x) }, { MULX((a).y, (b).y) }, { MULX((a).z, (b).z) } })
#define VMULNX(a, b) ((vf32x){ { MULX((a).x, (b)) }, { MULX((a).y, (b)) }, { MULX((a).z, (b)) } })
#define VSUBX(a, b) ((vf32x){ { SUBX((a).x, (b).x) }, { SUBX((a).y, (b).y) }, { SUBX((a).z, (b).z) } })
#define VDIVX(a, b) ((vf32x){ { DIVX((a).x, (b).x) }, { DIVX((a).y, (b).y) }, { DIVX((a).z, (b).z) } })
#define VDIVNX(a, b) ((vf32x){ { DIVX((a).x, (b)) }, { DIVX((a).y, (b)) }, { DIVX((a).z, (b)) } })
#define VFMAX(a, b, c) ((vf32x){ { FMAX((a).x, (b).x, (c).x) }, { FMAX((a).y, (b).y, (c).y) }, { FMAX((a).z, (b).z, (c).z) } })
#define VFMANX(a, b, c) ((vf32x){ { FMAX((a).x, (b), (c).x) }, { FMAX((a).y, (b), (c).y) }, { FMAX((a).z, (b), (c).z) } })
#define VDOTX(a, b) FMAX((a).x, (b).x, FMAX((a).y, (b).y, MULX((a).z, (b).z)))
#define VDOTNX__(a, b) FMAX((a).x, b##x, FMAX((a).y, b##y, MULX((a).z, b##z)))
#define VLENGTHX(a) SQRTX(FMAX((a).x, (a).x, FMAX((a).y, (a).y, MULX((a).z, (a).z))))
vf32x vnormalizex(vf32x a) {
    f32x reciprocal = SQRTX(VDOTX(a, a)); // TODO: RSQRTX doesn't work as expected here on NEON
    return (vf32x){ { DIVX(a.x, reciprocal) }, { DIVX(a.y, reciprocal) }, { DIVX(a.z, reciprocal) } };
}
#define VNORMALIZEX(a) vnormalizex(a)
#define VLERPX(a, b, t) VADDX(VMULNX((a), SUBX(F32NX(1.0f), (t))), VMULNX((b), (t)))

static inline vf32x VGATHERX_F32(void const * base0, void const * base1, void const * base2, u32 stride, u32x indices) {
    return ((vf32x){
        { GATHERX_F32(base0, stride, indices) },
        { GATHERX_F32(base1, stride, indices) },
        { GATHERX_F32(base2, stride, indices) }
    });
}

static inline vf32x VCHOOSEX(vf32x a, vf32x b, u32x t) {
    return ((vf32x){
        { CHOOSEX(a.x, b.x, t) },
        { CHOOSEX(a.y, b.y, t) },
        { CHOOSEX(a.z, b.z, t) }
    });
}

// https://prng.di.unimi.it/xoshiro128plus.c
static inline u32x wxoshiro128plus(u32x * s) {
    u32x result = ADDX(s[0], s[3]);

    u32x t = SLLX(s[1], 9);

    s[2] = XORX(s[2], s[0]);
    s[3] = XORX(s[3], s[1]);
    s[1] = XORX(s[1], s[2]);
    s[0] = XORX(s[0], s[3]);

    s[2] = XORX(s[2], t);

    s[3] = ORX(SLLX(s[3], 11), SRLX(s[3], 21));

    return result;
}

// https://prng.di.unimi.it/xoshiro128plus.c
static inline u32 xoshiro128plus(u32 * s) {
    u32 result = s[0] + s[3];

    u32 t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = (s[3] << 11) | (s[3] >> 21);

    return result;
}

u32x ws[4];
u32 s[4];

#define U32X_EXTRACT_F32X(a) MULX(U32X_TO_F32X(SRLX(a, 8)), F32NX(0x1.0p-24f))

#define U32_EXTRACT_F32(a) ((f32)(a >> 8) * 0x1.0p-24f)

static inline f32x wr01() {
    return U32X_EXTRACT_F32X(wxoshiro128plus(ws));
}

static inline f32 r01() {
    return U32_EXTRACT_F32(xoshiro128plus(s));
}

static inline f32x wr() {
    return SUBX(MULX(F32NX(2.0f), U32X_EXTRACT_F32X(wxoshiro128plus(ws))), F32NX(1.0f));
}

static inline f32 r() {
    return 2.0f * U32_EXTRACT_F32(xoshiro128plus(s)) - 1.0f;
}

static inline void random_disk(f32x * rx, f32x * ry) {
    f32x r = SQRTX(wr01());
    f32x theta = MULX(F32NX(F32_PI), wr());

    f32x sin_theta;
    f32x cos_theta;
    SINCOSX(theta, &sin_theta, &cos_theta);

    *rx = MULX(r, cos_theta);
    *ry = MULX(r, sin_theta);
}

static inline vf32x sphere_random(void) {
    // https://mathworld.wolfram.com/SpherePointPicking.html
    f32x u = wr();
    f32x v = wr01();

    f32x theta = MULX(F32NX(F32_PI), u);
    f32x phi = ACOSX(SUBX(MULX(F32NX(2.0f), v), F32NX(1.0f)));

    f32x sin_theta;
    f32x cos_theta;
    SINCOSX(theta, &sin_theta, &cos_theta);

    f32x sin_phi;
    f32x cos_phi;
    SINCOSX(phi, &sin_phi, &cos_phi);

    return (vf32x){
        { MULX(cos_phi, sin_theta) },
        { MULX(sin_phi, sin_theta) },
        { cos_theta }
    };
}

static inline u32x ray_sphere_intersection(f32x t, vf32x r, vf32x rd, struct Spheres const * spheres, u32 sphere_index, f32x * intersection) {
    vf32x s0 = VF32NX(spheres->x0[sphere_index], spheres->y0[sphere_index], spheres->z0[sphere_index]);
    vf32x s1 = VF32NX(spheres->x1[sphere_index], spheres->y1[sphere_index], spheres->z1[sphere_index]);
    vf32x s = VLERPX(s0, s1, t);

    f32x sr = F32NX(spheres->r[sphere_index]);
    f32x sr2 = MULX(sr, sr);

    vf32x o = VSUBX(r, s);

    f32x h = VDOTX(o, rd);
    f32x c = SUBX(VDOTX(o, o), sr2);

    f32x discriminant = SUBX(MULX(h, h), c);
    u32x discriminant_mask = GTX(discriminant, F32NX(0.0f));
    if (ALL_ZEROS_X(discriminant_mask)) {
        return U32NX(0);
    }

    f32x discriminant_root = SQRTX(discriminant);

    f32x first_root = SUBX(NEGX(h), discriminant_root);
    u32x first_root_mask = GTX(first_root, F32NX(0.001f));

    // We can probably do an early return here as well, altough we don't
    // really save much computation since _most_ hits will be the first
    // root. We only get the second root if we're inside the sphere(?).

    f32x second_root = SUBX(discriminant_root, h);

    // We know that first_root < second_root, but we need to make sure
    // that first_root is positive, otherwise we want the second root.
    f32x root = CHOOSEX(first_root, second_root, NOTX(first_root_mask));

    *intersection = root;

    u32x mask = ANDX(GTX(root, F32NX(0.001f)), discriminant_mask);
    return mask;
}

#define MAX_DEPTH 50

static inline void sphere_uv(f32x x, f32x y, f32x z, f32x * u, f32x * v) {
    f32x phi = ADDX(ATAN2X(NEGX(z), x), F32NX(F32_PI));
    f32x theta = ACOSX(NEGX(y));

    *u = DIVX(phi, F32NX(F32_2_PI));
    *v = DIVX(theta, F32NX(F32_PI));
}

static u64 rays_computed = 0;
static u64 rays_utilised = 0;

static inline u32 compute_pixel(u32 x, u32 y, u32 width, u32 height, u32 samples_count, f32 lens_radius, f32x4 look_u, f32x4 look_v, f32x4 origin, f32x4 horizontal, f32x4 vertical, f32 t0, f32 t1, struct Spheres const * spheres, u32 spheres_count) {
    // We keep this as a wide register even though we don't need to
    // to avoid having to do horizontal adds for each sample and only
    // do one horizontal add at the end.
    vf32x trgb = VF32NX(0.0f, 0.0f, 0.0f);

    for (u32 sample = 0; sample < samples_count / SIMD_WIDTH; sample += 1) {
        f32x trr = wr();
        f32x t = ADDX(MULX(SUBX(F32NX(1.0f), trr), F32NX(t0)), MULX(trr, F32NX(t1)));

        f32x u = SUBX(MULX(F32NX(2.0f), DIVX(ADDX(F32NX(x), wr()), F32NX(width - 1))), F32NX(1.0f));
        f32x v = SUBX(MULX(F32NX(2.0f), DIVX(ADDX(F32NX(y), wr()), F32NX(height - 1))), F32NX(1.0f));

        // "Defocus blur" or depth of field
        #if 0
            f32x rdox;
            f32x rdoy;
            random_disk(&rdox, &rdoy);
            f32x rox = MULX(rdox, F32NX(lens_radius));
            f32x roy = MULX(rdoy, F32NX(lens_radius));
        #else
            f32x rox = F32NX(0.0f);
            f32x roy = F32NX(0.0f);
        #endif

        vf32x o = VFMANX(VF32NX(look_u.x, look_u.y, look_u.z), rox, VMULNX(VF32NX(look_v.x, look_v.y, look_v.z), roy));

        vf32x rp = VADDX(VF32NX(origin.x, origin.y, origin.z), o);

        vf32x rd = VSUBX(VFMANX(VF32NX(horizontal.x, horizontal.y, horizontal.z), u, VMULNX(VF32NX(vertical.x, vertical.y, vertical.z), v)), rp);
        rd = VNORMALIZEX(rd);

        vf32x color = VF32NX(1.0f, 1.0f, 1.0f);

        u32x active = U32NX(0xFFFFFFFF);
        for (u32 depth = 0; depth < MAX_DEPTH; depth += 1) {
            f32x intersection = F32NX(F32_INFINITY);

            u32x candidate_sphere_index = U32NX(0); // i.e. _which_ sphere did we hit?
            u32x candidate_sphere_index_predicate = U32NX(0); // i.e. did we hit _any_ sphere?
            for (u32 sphere_index = 0; sphere_index < spheres_count; sphere_index += 1) {
                f32x candidate_intersection = F32NX(F32_INFINITY);
                u32x result = ray_sphere_intersection(t, rp, rd, spheres, sphere_index, &candidate_intersection);
                //
                // It looks like clang gets royally confused here. The following if-statement _should_ not be required
                // as we short circuit the ray_sphere_intersection call if all determinants are < 0. But not having this
                // if statement and using & when computing the predicate it ~2-3x slower than using &&. I think this is
                // because && allows the computation of the predicate to be short circuited, and therefore allow us to
                // skip computation. I don't really see how clang gets into this situation, but it's some optimization
                // shenanigans. Adding this if-statement doesn't have any logical effect what so ever, but allows the
                // use of & instead of && which makes SIMD implementation easier.
                //
                if (!ALL_ZEROS_X(result)) {
                    u32x intersection_predicate = ANDX(result, LTX(candidate_intersection, intersection));
                    intersection = CHOOSEX(intersection, candidate_intersection, intersection_predicate);
                    candidate_sphere_index = CHOOSEX_U32(candidate_sphere_index, U32NX(sphere_index), intersection_predicate);
                    candidate_sphere_index_predicate = ORX(candidate_sphere_index_predicate, intersection_predicate);
                }
            }

            vf32x candidate = VF32NX(0.0f, 0.0f, 0.0f);

            if (!ALL_ZEROS_X(candidate_sphere_index_predicate)) {
                vf32x p = VFMAX(VF32X(intersection, intersection, intersection), rd, rp);

                vf32x so0 = VGATHERX_F32(spheres->x0, spheres->y0, spheres->z0, sizeof(f32), candidate_sphere_index);
                vf32x so1 = VGATHERX_F32(spheres->x1, spheres->y1, spheres->z1, sizeof(f32), candidate_sphere_index);
                // (1 - t) * a + t * b
                vf32x so = VLERPX(so0, so1, t);
                f32x sr = GATHERX_F32(spheres->r, sizeof(f32), candidate_sphere_index);

                vf32x n = VDIVNX(VSUBX(p, so), sr);

                u32x front  = LTX(VDOTX(rd, n), F32NX(0.0f));

                vf32x dn = VCHOOSEX(VNEGX(n), n, front);

                rp = p;

                f32x refractive_index = GATHERX_F32(spheres->material.dielectric.refractive_index, sizeof(u32), candidate_sphere_index);
                f32x roughness = GATHERX_F32(spheres->material.metal.roughness, sizeof(f32), candidate_sphere_index);

                vf32x albedo = VGATHERX_F32(spheres->material.albedo.r, spheres->material.albedo.g, spheres->material.albedo.b, sizeof(f32), candidate_sphere_index);

                // TODO: The ABSX here causes the texture to repeat around the 0, making the checkers
                //       there larger.
                u32x even = EQX(ANDX(ADDX(F32X_TO_U32X(ABSX(MULX(p.x, F32NX(2.0f)))), F32X_TO_U32X(ABSX(MULX(p.z, F32NX(2.0f))))), U32NX(1)), U32NX(0));
                vf32x texture = VCHOOSEX(VF32NX(0.0f, 0.0f, 0.0f), VF32NX(1.0f, 1.0f, 1.0f), even);

                u32x use_texture = GATHERX_U32(spheres->material.use_texture, sizeof(u32), candidate_sphere_index);

                vf32x material_color = VCHOOSEX(albedo, texture, use_texture);

                u32x material_type = GATHERX_U32(spheres->material.type, sizeof(u32), candidate_sphere_index);

                // This assumes that the other material is vacuum, but we should keep track of the other material and use that here
                f32x inverse_dielec_refractive_ratio = DIVX(F32NX(1.0f), refractive_index);
                f32x dielec_refractive_ratio = CHOOSEX(refractive_index, inverse_dielec_refractive_ratio, front);

                vf32x rs = sphere_random();

                vf32x lamb_scattering_direction = VADDX(n, rs);

                u32x lamb_scattering_predicate = ANDX(ANDX(LTX(ABSX(lamb_scattering_direction.x), F32NX(1E-3f)), LTX(ABSX(lamb_scattering_direction.y), F32NX(1E-3f))), LTX(ABSX(lamb_scattering_direction.z), F32NX(1E-3)));
                lamb_scattering_direction = VCHOOSEX(lamb_scattering_direction, n, lamb_scattering_predicate);
                // I don't think we need to do this, is it not already normal?
                lamb_scattering_direction = VNORMALIZEX(lamb_scattering_direction);

                f32x metal_reflection_d = MULX(F32NX(2.0f), VDOTX(rd, dn));
                vf32x metal_reflection = VSUBX(rd, VMULNX(dn, metal_reflection_d));
                vf32x metal_reflection2 = VFMANX(rs, roughness, metal_reflection);
                // I don't think we need to do this, is it not already normal?
                vf32x metal_new_rd = VNORMALIZEX(metal_reflection2);

                vf32x nrd = VNEGX(rd);

                // We calculate some of these values inside refract, should we inline refract here?
                f32x dielec_cos_theta = MINX(VDOTX(nrd, dn), F32NX(1.0f));
                f32x dielec_sin_theta = SQRTX(SUBX(F32NX(1.0f), MULX(dielec_cos_theta, dielec_cos_theta))); // Might it be faster to calculate this without using the trigonometric identity?

                u32x dielec_total_internal_reflection = GTX(MULX(dielec_refractive_ratio, dielec_sin_theta), F32NX(1.0f));

                // Calculate Schlick's approximation
                f32x schlick;
                {
                    f32x r0 = DIVX(SUBX(F32NX(1.0f), dielec_refractive_ratio), ADDX(F32NX(1.0f), dielec_refractive_ratio));
                    f32x r02 = MULX(r0, r0);
                    schlick = FMAX(SUBX(F32NX(1.0f), r02), POW5X(SUBX(F32NX(1.0f), dielec_cos_theta)), r02);
                }
                u32x dielec_schlick = GTX(schlick, wr01());

                // Total internal reflection, there's no solution to Snell's law:
                //   sin'(theta) = n / n' * sin(theta)
                // Where n / n' > 1
                u32x dielec_do_reflection = ORX(dielec_total_internal_reflection, dielec_schlick);

                // Reflect
                // i - n * 2 * dot(i, n)
                vf32x dielec_reflection = metal_reflection;

                // Refract
                vf32x dielec_refraction;
                {
                    f32x perpm = MINX(VDOTX(nrd, dn), F32NX(1.0f));
                    vf32x perpendicular = VMULNX(VADDX(rd, VMULNX(dn, perpm)), dielec_refractive_ratio);

                    // I don't think we need to do this, is it not already normal?
                    f32x p = NEGX(SQRTX(ABSX(SUBX(F32NX(1.0f), VDOTX(perpendicular, perpendicular)))));
                    vf32x parallel = VMULNX(dn, p);

                    dielec_refraction = VADDX(perpendicular, parallel);
                }

                vf32x dielec_reflection_or_refraction = VCHOOSEX(dielec_refraction, dielec_reflection, dielec_do_reflection);
                // I don't think we need to do this, is it not already normal?
                dielec_reflection_or_refraction = VNORMALIZEX(dielec_reflection_or_refraction);

                {
                    u32x predicate = ORX(EQX(material_type, U32NX(LAMBERTIAN)), EQX(material_type, U32NX(METAL)));
                    candidate = VMULX(color, VCHOOSEX(VF32NX(1.0f, 1.0f, 1.0f), material_color, predicate));
                }

                vf32x trd = VCHOOSEX(metal_new_rd, lamb_scattering_direction, EQX(material_type, U32NX(LAMBERTIAN)));

                rd = VCHOOSEX(trd, dielec_reflection_or_refraction, EQX(material_type, U32NX(DIELECTRIC)));
            }

            f32x y = FMAX(F32NX(0.5f), rd.y, F32NX(0.5f));
            vf32x color2 = VMULNX(color, FMAX(F32NX(0.5f), y, SUBX(F32NX(1.0f), y)));

            candidate = VCHOOSEX(candidate, color2, EQX(candidate_sphere_index_predicate, U32NX(0)));

            color = VCHOOSEX(color, candidate, EQX(active, U32NX(0xFFFFFFFF)));

            // We can avoid the horizontal add here if we store the counts locally to
            // the thread in a wide register and only do the summation at the very
            // end of the thread.
            __atomic_add_fetch(&rays_computed, SIMD_WIDTH, __ATOMIC_SEQ_CST);
            __atomic_add_fetch(&rays_utilised, (u64)HADDX(ANDX(active, U32NX(1))), __ATOMIC_SEQ_CST);

            active = ANDX(active, NEQX(candidate_sphere_index_predicate, U32NX(0)));
            if (ALL_ZEROS_X(active)) {
                break;
            }
        }

        trgb = VADDX(trgb, color);
    }

    // Probably move the sRGB computation here? Makes sense since we have
    // the values and we can do it wide.
    f32 ttr = HADDX(trgb.r) / samples_count;
    f32 ttg = HADDX(trgb.g) / samples_count;
    f32 ttb = HADDX(trgb.b) / samples_count;

    // We need a ARM NEON pow function to be able to do with wide, and we might want to do it for 4 pixels at a time?
    // https://en.wikipedia.org/wiki/SRGB
    // https://entropymine.com/imageworsener/srgbformula/
    f32 r = ttr;
    if (r <= 0.00313066844250063f) {
        r = 12.92f * r;
    } else {
        r = 1.055f * powf(r, 1.0f / 2.4f) - 0.055f;
    }

    f32 g = ttg;
    if (g <= 0.00313066844250063f) {
        g = 12.92f * g;
    } else {
        g = 1.055f * powf(g, 1.0f / 2.4f) - 0.055f;
    }

    f32 b = ttb;
    if (b <= 0.00313066844250063f) {
        b = 12.92f * b;
    } else {
        b = 1.055f * powf(b, 1.0f / 2.4f) - 0.055f;
    }

    u8 ru = (u8)(255.0f * r);
    u8 gu = (u8)(255.0f * g);
    u8 bu = (u8)(255.0f * b);

    u32 result = ((u32)ru << 24) | ((u32)gu << 16) | ((u32)bu << 8);
    return result;
}

static inline f32 degrees_to_radians(f32 degrees) {
    return degrees * (F32_PI / 180.0f);
}

static inline f32x4 subtract(f32x4 a, f32x4 b) {
    return (f32x4){ a.x - b.x, a.y - b.y, a.z - b.z };
}

static inline f32x4 multiply(f32x4 a, f32 b) {
    return (f32x4){ a.x * b, a.y * b, a.z * b };
}

static inline f32 dot(f32x4 a, f32x4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline f32x4 cross(f32x4 a, f32x4 b) {
    return (f32x4){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static inline f32 distance(f32x4 a, f32x4 b) {
    f32x4 d = subtract(b, a);
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

static inline f32x4 normalize(f32x4 a) {
    f32 l = sqrtf(dot(a, a));
    return (f32x4){
        a.x / l,
        a.y / l,
        a.z / l,
    };
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void compute_tile(struct Job * job) {
    for (u32 y = job->from_y; y < job->to_y + 1; y += 1) {
        for (u32 x = job->from_x; x < job->to_x + 1; x += 1) {
            // compute_pixel should return a vector register so we can do a vector store here
            u32 pixel = compute_pixel(x, y, job->total_width, job->total_height, job->samples_count, job->lens_radius, job->look_u, job->look_v, job->origin, job->horizontal, job->vertical, job->t0, job->t1, job->spheres, job->spheres_count);

            u32 width = (job->to_x + 1) - job->from_x;
            u32 height = (job->to_y + 1) - job->from_y;
            job->buffer[(y - job->from_y) * 3 * width + (x - job->from_x) * 3 + 0] = (u8)((pixel >> 24) & 0xFF);
            job->buffer[(y - job->from_y) * 3 * width + (x - job->from_x) * 3 + 1] = (u8)((pixel >> 16) & 0xFF);
            job->buffer[(y - job->from_y) * 3 * width + (x - job->from_x) * 3 + 2] = (u8)((pixel >> 8) & 0xFF);
        }
    }
}

struct Worker {
    u32 worker_index;
    struct Job_Queue * job_queue;
};

void * do_jobs(void * pointer) {
    struct Worker * worker = (struct Worker *)pointer;

    struct Job_Queue * job_queue = worker->job_queue;

    u32 job_index = __atomic_add_fetch(&job_queue->jobs_index, 1, __ATOMIC_SEQ_CST) - 1;
    while (job_index < job_queue->jobs_count) {
        struct Job * const job = &job_queue->jobs[job_index];

        compute_tile(job);

        if (worker->worker_index == 0) {
            printf("\r%" PRIu32 " %%", (u32)(100.0f * ((f32)job_index / (f32)(job_queue->jobs_count - 1))));
            fflush(stdout);
        }

        job_index = __atomic_add_fetch(&job_queue->jobs_index, 1, __ATOMIC_SEQ_CST) - 1;
    }

    return NULL;
}

struct Material_Storage {
    u32 type;

    struct {
        f32 r;
        f32 g;
        f32 b;
    } albedo;

    struct {
        f32 roughness;
    } metal;

    struct {
        f32 refractive_index;
    } dielectric;

    u32 use_texture;
} material;

void add_sphere(struct Spheres * spheres, u32 index, f32 x, f32 y, f32 z, f32 r, struct Material_Storage material) {
    spheres->x0[index] = x;
    spheres->x1[index] = x;
    spheres->y0[index] = y;
    spheres->y1[index] = y;
    spheres->z0[index] = z;
    spheres->z1[index] = z;
    spheres->r[index] = r;
    spheres->material.type[index] = material.type;
    spheres->material.albedo.r[index] = material.albedo.r;
    spheres->material.albedo.g[index] = material.albedo.g;
    spheres->material.albedo.b[index] = material.albedo.b;
    spheres->material.metal.roughness[index] = material.metal.roughness;
    spheres->material.dielectric.refractive_index[index] = material.dielectric.refractive_index;
    spheres->material.use_texture[index] = material.use_texture;
}

void add_interpolating_sphere(struct Spheres * spheres, u32 index, f32 x0, f32 x1, f32 y0, f32 y1, f32 z0, f32 z1, f32 r, struct Material_Storage material) {
    spheres->x0[index] = x0;
    spheres->x1[index] = x1;
    spheres->y0[index] = y0;
    spheres->y1[index] = y1;
    spheres->z0[index] = z0;
    spheres->z1[index] = z1;
    spheres->r[index] = r;
    spheres->material.type[index] = material.type;
    spheres->material.albedo.r[index] = material.albedo.r;
    spheres->material.albedo.g[index] = material.albedo.g;
    spheres->material.albedo.b[index] = material.albedo.b;
    spheres->material.metal.roughness[index] = material.metal.roughness;
    spheres->material.dielectric.refractive_index[index] = material.dielectric.refractive_index;
    spheres->material.use_texture[index] = material.use_texture;
}

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

int main(int argc, char ** argv) {
    u64 begin = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);

    struct Spheres * spheres = malloc(sizeof(struct Spheres));

    add_sphere(spheres, 0, 0.0f, -1000.0f, 0.0f, 1000.0f, (struct Material_Storage){
        .type = LAMBERTIAN,
        .albedo = { 0.5f, 0.5f, 0.5f },
        .use_texture = 0xFFFFFFFF,
    });

    add_sphere(spheres, 1, 0.0f, 1.0f, 0.0f, 1.0f, (struct Material_Storage){
        .type = DIELECTRIC,
        .dielectric = {
            .refractive_index = 1.5f
        },
        .use_texture = 0
    });

     add_sphere(spheres, 2, -4.0f, 1.0f, 0.0f, 1.0f, (struct Material_Storage){
        .type = LAMBERTIAN,
        .albedo = { 0.4f, 0.2f, 0.1f },
        .use_texture = 0
    });

     add_sphere(spheres, 3, 4.0f, 1.0f, 0.0f, 1.0f,  (struct Material_Storage){
        .type = METAL,
        .albedo = { 0.7f, 0.6f, 0.5f },
        .metal = {
            .roughness = 0.0f
        },
        .use_texture = 0
    });

    // Seed the random number generator with high quality entropy.
    // https://support.apple.com/en-gb/guide/security/seca0c73a75b/web
    getentropy(s, (u32)sizeof(s));
    getentropy(ws, (u32)sizeof(ws));

    u32 spheres_count = 4;
    for (u32 a = 0; a < 22; a += 1) {
        for (u32 b = 0; b < 22; b += 1) {
            f32x4 position0 = (f32x4){ ((f32)a - 11.0f) + 0.9f * r(), 0.2f, ((f32)b - 11.0f) + 0.9f * r() };
            f32x4 position1 = (f32x4){ position0.x, position0.y + ((r01() < 0.6f) ? 0.2f * r01() : 0.0f), position0.z };

            if (distance(position0, (f32x4){ 4.0f, 0.2f, 0.0f }) > 0.9f) {
                struct Material_Storage material;

                f32 random_material = r01();
                if (random_material < 0.8f) {
                    // Lambertian
                    material = (struct Material_Storage){
                        .type = LAMBERTIAN,
                        .albedo = { r01() * r01(), r01() * r01(), r01() * r01() },
                        .use_texture = 0
                    };
                } else if (random_material < 0.95f) {
                    // Metal
                    material = (struct Material_Storage){
                        .type = METAL,
                        .albedo = { 0.5f + 0.5f * r01(), 0.5f + 0.5f * r01(), 0.5f + 0.5f * r01() },
                        .metal = {
                            .roughness = 0.5f * r01()
                        },
                        .use_texture = 0
                    };
                } else {
                    // Dielectric
                    material = (struct Material_Storage){
                        .type = DIELECTRIC,
                        .dielectric = {
                            .refractive_index = 1.5f
                        },
                        .use_texture = 0
                    };
                }

                add_interpolating_sphere(spheres, spheres_count, position0.x, position1.x, position0.y, position1.y, position0.z, position1.z, 0.2f, material);
                spheres_count += 1;
            }
        }
    }

    f32 aspect = 16.0f / 9.0f;
    u32 width = 1200;
    u32 height = (u32)((f32)width / aspect);

    f32 vertical_field_of_view = degrees_to_radians(20.0f);

    f32 viewport_height = 2.0f * tanf(vertical_field_of_view / 2.0f);
    f32 viewport_width = aspect * viewport_height;

    f32x4 look_from = (f32x4){ 13.0f, 2.0f, 3.0f };
    f32x4 look_to = (f32x4){ 0.0f, 0.0f, 0.0f };
    f32x4 look_up = (f32x4){ 0.0f, 1.0f, 0.0f };

    f32 aperture = 0.1f;
    f32 focus_distance = 10.0f;
    f32 lens_radius = aperture / 2.0f;

    f32x4 look_w = normalize(subtract(look_from, look_to));
    f32x4 look_u = normalize(cross(look_up, look_w));
    f32x4 look_v = normalize(cross(look_w, look_u));

    f32x4 origin = look_from;
    f32x4 horizontal = multiply(look_u, focus_distance * viewport_width / 2.0f);
    f32x4 vertical = multiply(look_v, focus_distance * viewport_height / 2.0f);

    u32 const samples_count = 16;
    _Static_assert(samples_count % SIMD_WIDTH == 0, "Samples count must be a multiple of the SIMD width");

    // The Apple M1 Max has 10 cores; 8 performance cores and 2 efficiency cores. We
    // get better performance with 8 threads instead of 10, around 2 seconds with 8
    // cores compared to 2.5 seconds with 10 cores.
    u32 thread_concurrency = 8;
    u32 simd_concurrency = SIMD_WIDTH;

    u32 tile_width = 64;
    u32 tile_height = 64;

    u32 tiles_count_x = (width + tile_width - 1) / tile_width;
    u32 tiles_count_y = (height + tile_height - 1) / tile_height;
    u32 tiles_count = tiles_count_x * tiles_count_y;

    printf("Concurrency: %" PRIu32 " threads, %" PRIu32 "-wide\n", thread_concurrency, simd_concurrency);
    printf("Resolution: %" PRIu32 "x%" PRIu32 " with %" PRIu32 " samples per pixel, divided into %" PRIu32 " tiles á %" PRIu32 "x%" PRIu32 "\n", width, height, samples_count, tiles_count, tile_width, tile_height);

    u32 jobs_count = tiles_count;
    struct Job_Queue job_queue = (struct Job_Queue){
        .jobs = malloc(jobs_count * sizeof(struct Job)),
        .jobs_count = jobs_count,
        .jobs_index = 0
    };

    for (u32 tile_y = 0; tile_y < tiles_count_y; tile_y += 1) {
        for (u32 tile_x = 0; tile_x < tiles_count_x; tile_x += 1) {
            u32 tile_index = (tile_y * tiles_count_x) + tile_x;

            u8 * buffer = malloc(tile_width * tile_height * 3);

            u32 job_index = tile_index;
            job_queue.jobs[job_index] = (struct Job){
                .from_x = tile_x * tile_width,
                .from_y = tile_y * tile_height,
                .to_x = MIN((tile_x + 1) * tile_width - 1, width - 1),
                .to_y = MIN((tile_y + 1) * tile_height - 1, height - 1),

                .total_width = width,
                .total_height = height,
                .buffer = buffer,

                .samples_count = samples_count,
                .lens_radius = lens_radius,
                .look_u = look_u,
                .look_v = look_v,
                .origin = origin,
                .horizontal = horizontal,
                .vertical = vertical,
                .t0 = 0.0f,
                .t1 = 1.0f,

                .spheres = spheres,
                .spheres_count = spheres_count,
            };
        }
    }

    struct Worker * workers = malloc(thread_concurrency * sizeof(struct Worker));
    for (u32 worker_index = 0; worker_index < thread_concurrency; worker_index += 1) {
        workers[worker_index] = (struct Worker) {
            .worker_index = worker_index,
            .job_queue = &job_queue,
        };
    }

    pthread_t * threads = malloc(thread_concurrency * sizeof(pthread_t));
    for (u32 thread_index = 1; thread_index < thread_concurrency; thread_index += 1) {
        pthread_create(&threads[thread_index], NULL, do_jobs, &workers[thread_index]);
    }

    do_jobs(&workers[0]);

    for (u32 thread_index = 1; thread_index < thread_concurrency; thread_index += 1) {
        pthread_join(threads[thread_index], NULL);
    }

    u8 * final_buffer = malloc(width * height * 3);
    for (u32 tile_y = 0; tile_y < tiles_count_y; tile_y += 1) {
        for (u32 tile_x = 0; tile_x < tiles_count_x; tile_x += 1) {
            u32 tile_index = (tile_y * tiles_count_x) + tile_x;

            u32 job_index = tile_index;
            struct Job * job = &job_queue.jobs[job_index];

            u32 job_width = (job->to_x + 1) - job->from_x;
            u32 job_height = (job->to_y + 1) - job->from_y;
            for (u32 y = job->from_y; y < job->to_y + 1; y += 1) {
                for (u32 x = job->from_x; x < job->to_x + 1; x += 1) {
                    final_buffer[(y * job->total_width * 3) + (x * 3) + 0] = job->buffer[((y - job->from_y) * job_width * 3) + ((x - job->from_x) * 3) + 0];
                    final_buffer[(y * job->total_width * 3) + (x * 3) + 1] = job->buffer[((y - job->from_y) * job_width * 3) + ((x - job->from_x) * 3) + 1];
                    final_buffer[(y * job->total_width * 3) + (x * 3) + 2] = job->buffer[((y - job->from_y) * job_width * 3) + ((x - job->from_x) * 3) + 2];
                }
            }
        }
    }

    u64 end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    u64 elapsed = end - begin;

    printf("\n");

    printf("Elapsed time: %.04f s\n", (f64)elapsed / 1E9);
    printf("  Number of rays computed: %" PRIu64 "\n", rays_computed);
    printf("  Number of rays utilised: %" PRIu64 " (%" PRIu64 "%% utilisation)\n", rays_utilised, (u64)(100.0 * ((f64)rays_utilised / (f64)rays_computed)));
    printf("  Per ray: %.06f μs\n", ((f64)elapsed / 1E3) / (f64)rays_computed);

    char file_name[128];
    snprintf(file_name, sizeof(file_name), "%s.png", (argc > 1 ? argv[1] : "rt"));

    stbi_flip_vertically_on_write(1);
    stbi_write_png(file_name, width, height, 3, final_buffer, width * 3);

    return 0;
}
