struct box {
    float2 tl;
    float2 tr;
    float2 br;
    float2 bl;
};

struct line {
    float2 a;
    float2 b;
};

float cross2(float2 a, float2 b) {
    return a[0] * b[1] - a[1] * b[0];
}

float2 lineLineIntersection(struct line l1, struct line l2) {
    float2 a = l1.a;
    float2 b = l1.b;
    float2 c = l2.a;
    float2 d = l2.b;

    float2 r = b - a;
    float2 s = d - c;

    float denom = cross2(r, s);
    if (denom == 0) return (float2){INFINITY, INFINITY};

    float t = cross2(c - a, s) / denom;
    return a + r * t;
}

struct box boxFromDataRepr(__global float* data) {
    float half_w = data[2] * data[2] / 2.0f;
    float half_h = data[3] * data[3] / 2.0f;
    float r = data[4];

    float2 center = {data[0], data[1]};
    float2 x_axis = {cos(r), sin(r)};
    float2 y_axis = {-x_axis[1], x_axis[0]};

    float2 x_edge_offs = (half_w * x_axis);
    float2 y_edge_offs = (half_h * y_axis);

    // Consider top left 0, 0
    float2 tl = center - x_edge_offs - y_edge_offs;
    float2 tr = center + x_edge_offs - y_edge_offs;
    float2 br = center + x_edge_offs + y_edge_offs;
    float2 bl = center - x_edge_offs + y_edge_offs;

    return (struct box) {
        .tl = tl,
        .tr = tr,
        .bl = bl,
        .br = br,
    };
}

struct intersection_polygon { float2 data[8]; };

struct polygon_builder {
    float2 data[8];
    unsigned char len;
};

void polygon_builder_init(struct polygon_builder* builder) {
    for (int i = 0; i < 8; i++) {
        builder->data[i] = INFINITY;
    }
    builder->len = 0;
}

void polygon_builder_push(struct polygon_builder* builder, float2 val) {
    builder->data[builder->len] = val;
    builder->len++;
}

bool shouldClipVertex(struct line line, float2 p) {
    float2 ab = line.b - line.a;
    float2 rot_90_axis = normalize((float2){ ab[1], -ab[0] });

    float2 ap = p - line.a;
    return dot(ap, rot_90_axis) > 0;
}

bool pointInBounds(float2 point, struct line line) {
    float2 ap = point - line.a;
    float2 ab = line.b - line.a;
    float ab_len = length(ab);
    float2 ab_norm = ab / ab_len;
    float d = dot(ap, ab_norm);
    return d <= ab_len && d >= 0;
}

struct intersection_polygon calcIntersection(struct box b1, struct box b2) {

    struct polygon_builder builders[2];

    polygon_builder_init(&builders[0]);
    polygon_builder_push(&builders[0], b2.tl);
    polygon_builder_push(&builders[0], b2.tr);
    polygon_builder_push(&builders[0], b2.br);
    polygon_builder_push(&builders[0], b2.bl);

    unsigned char out_idx = 0;
    float2 clip_points[4] = {b1.tl, b1.tr, b1.br, b1.bl };

    for (int clip_idx = 0; clip_idx < 4; clip_idx++) {
        struct polygon_builder *in = &builders[out_idx];

        out_idx = out_idx ^ 1;
        struct polygon_builder *out = &builders[out_idx];

        polygon_builder_init(out);

        struct line clip_edge = {clip_points[clip_idx], clip_points[(clip_idx + 1) % 4]};
        for (unsigned char i = 0; i < in->len; i++ ){
            float2 p = in->data[i];
            if (!shouldClipVertex(clip_edge, p)) {
                polygon_builder_push(out, p);
            }

            float2 next_p = in->data[(i + 1) % in->len];
            struct line intersection_line = { p, next_p };
            float2 intersection_point = lineLineIntersection(clip_edge, intersection_line);
            if (pointInBounds(intersection_point, intersection_line)) {
                polygon_builder_push(out, intersection_point);
            }
        }
    }

    struct intersection_polygon ret;
    printf("out len %u\n", builders[out_idx].len);
    for (int i = 0; i < 8; i++) {
        ret.data[i] = builders[out_idx].data[i];
    }
    return ret;
}

__kernel void calc_iou(
    __global float* as,
    __global float* bs,
    __global float* out,
    uint n
) {
    // as, bs (5, N)
    // out (16, n)
    //
    // box repr is cx, cy, sqrt(w), sqrt(h), rot_rad

    uint global_id = get_global_id(0);
    if (global_id * 16 >= n) return;

    __global float* in_a = as + 5 * global_id;
    __global float* in_b = bs + 5 * global_id;

    struct box a_box = boxFromDataRepr(in_a);
    struct box b_box = boxFromDataRepr(in_b);

    struct intersection_polygon poly = calcIntersection(a_box, b_box);
    float2 intersection = lineLineIntersection((struct line){a_box.tl, a_box.tr}, (struct line){b_box.tl, b_box.tr});
    for (int i = 0; i < 8; i++) {
        out[global_id * 16 + i * 2] = poly.data[i][0];
        out[global_id * 16 + i * 2 + 1] = poly.data[i][1];
    }
}
