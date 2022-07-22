# shader vertex
# version 330 core

layout(location = 0) in vec3 vertices;
layout(location = 2) in vec2 texcoords;

uniform mat4 MVP;

out vec4 vertex_color;
out vec2 vertex_texcoord;

void main(){		
	gl_Position = MVP * vec4(vertices.x, vertices.y, vertices.z, 1.0f);
	vertex_texcoord = texcoords;
};

# shader fragment
# version 330 core

layout(location = 0) out vec4 color;

in vec4 vertex_color;
in vec2 vertex_texcoord;
uniform sampler2D texmap;

void main(){
	color = texture(texmap, vertex_texcoord);
};