// Script para mejorar la interactividad del sitio web
document.addEventListener('DOMContentLoaded', function() {
    // Añadir clase fade-in a elementos principales
    document.querySelectorAll('section').forEach(function(section) {
        section.classList.add('fade-in');
    });
    
    // Navegación suave
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Activar tooltips de Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Cambiar color de la barra de navegación al hacer scroll
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('bg-dark');
            navbar.classList.remove('bg-primary');
        } else {
            navbar.classList.add('bg-primary');
            navbar.classList.remove('bg-dark');
        }
    });
    
    // Añadir funcionalidad para mostrar más detalles en las tarjetas
    document.querySelectorAll('.card').forEach(function(card) {
        card.addEventListener('click', function() {
            // Aquí se podría implementar un modal o expandir la tarjeta
            // para mostrar más información
        });
    });
});
