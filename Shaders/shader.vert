#version 450

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 texCoord;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec2 coords[3] = vec2[](
    vec2(0.0, 0.0),
    vec2(1, 0.0),
    vec2(0, 1)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    vec4 p = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    gl_Position = ubo.proj * ubo.view * ubo.model * p;
    //gl_Position = ubo.model * p;
    fragColor = colors[gl_VertexIndex];
    texCoord = coords[gl_VertexIndex];
}