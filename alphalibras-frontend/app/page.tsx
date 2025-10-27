import { Card, CardHeader, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

import Image from "next/image";

export default function Home() {
  return (
    <div className="relative border-2 border-black rounded-md shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] bg-white
                    h-screen flex flex-col items-center justify-between overflow-hidden p-6

                    bg-[linear-gradient(to_right,theme(colors.gray.200)_1px,transparent_1px),linear-gradient(to_bottom,theme(colors.gray.200)_1px,transparent_1px)]
                    bg-[size:1.5rem_1.5rem]">
      
      {/* Cabeçalho */}
      <header className="flex w-full items-center justify-between p-6">
        <h1 className="flex items-center gap-2 text-2xl font-bold border-2 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] px-4 py-2 bg-white">
                    <Image
            src="/logo.png"
            alt="Logo AlphaLibras"
            width={28}
            height={28}
            className="h-7 w-7"
          />
          <span>AlphaLibras</span>

        </h1>
        <nav className="flex space-x-4">
          <Button className="text-xl font-bold">O que é Libras</Button>
          <Button className="text-xl font-bold">Sobre</Button>
        </nav>
      </header>

      {/* Área Central com os dois Cards */}
      <section className="flex flex-1 items-center justify-center gap-8 mb-8">
        
        <Card className="w-64 h-80 flex flex-col items-center justify-center gap-4 cursor-pointer hover:bg-gray-50 transition-colors">
          <Image
            src="/logo.png"
            alt="Ícone do Alfabeto em Libras"
            width={128}
            height={128}
          />
          <CardDescription className="text-xl text-center font-bold">
            Aprenda o Alfabeto
          </CardDescription>
        </Card>

        <Card className="w-64 h-80 flex flex-col items-center justify-center gap-4 cursor-pointer hover:bg-gray-50 transition-colors">
          <Image
            src="/logo.png"
            alt="Ícone do Alfabeto em Libras"
            width={128}
            height={128}
          />
          <CardDescription className="text-xl text-center font-bold">
            Soletre a Palavra
          </CardDescription>
        </Card>

      </section>

      {/* Rodapé com as imagens da instituição */}
      <footer className="w-full flex justify-end items-center space-x-4 p-6">
        <Image
          src="/brasao3_horizontal_cor_300dpi.png"
          alt="Universidade Federal do Ceará"
          width={3.75 * 30}
          height={30} 
          className="h-auto" 
        />
        <Image
          src="/logo-nuclic.png" 
          alt="Laboratório de Sistemas de Computação"
          width={3.75 * 30}
          height={30} 
          className="h-auto"
        />
      </footer>
    </div>
  );
}